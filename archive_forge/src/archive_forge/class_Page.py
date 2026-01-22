import os
from collections import namedtuple
import re
import sqlite3
import typing
import warnings
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects.packages_utils import (get_packagepath,
from collections import OrderedDict
class Page(object):
    """ An R documentation page.

    The original R structure is a nested sequence of components,
    corresponding to the latex-like .Rd file

    An help page is divided into sections, the names for the sections
    are the keys for the dict attribute 'sections', and a given section
    can be extracted with the square-bracket operator.

    In R, the S3 class 'Rd' is the closest entity to this class.
    """

    def __init__(self, struct_rdb: rinterface.ListSexpVector, _type: str=''):
        sections = OrderedDict()
        for elt_i in range(len(struct_rdb)):
            elt = rinterface.baseenv['['](struct_rdb, elt_i + 1)
            rd_tag = elt[0].do_slot('Rd_tag')[0]
            if rd_tag in sections and rd_tag not in NON_UNIQUE_TAGS:
                warnings.warn('Section of the R doc duplicated: %s' % rd_tag)
            sections[rd_tag] = elt
        self._sections = sections
        self._type = _type

    def _section_get(self):
        return self._sections
    sections = property(_section_get, None, None, 'Sections in the in help page, as a dict.')

    def __getitem__(self, item):
        """ Get a section """
        return self.sections[item]

    def arguments(self) -> typing.List[Item]:
        """ Get the arguments and descriptions as a list of Item objects. """
        section_doc = self._sections.get('\\arguments')
        res: typing.List[Item] = list()
        if section_doc is None:
            return res
        else:
            arg_name = None
            arg_desc = None
            section_rows = _Rd2txt(section_doc)
            if len(section_rows) < 3:
                return res
            for row in section_rows[2:]:
                if arg_name is None:
                    m = p_newarg.match(row)
                    if m:
                        arg_name = m.groups()[0]
                        arg_desc = [m.groups()[1]]
                elif p_desc.match(row):
                    arg_desc.append(row.strip())
                else:
                    res.append(Item(arg_name, arg_desc))
                    arg_name = None
                    arg_desc = None
            if arg_name is not None:
                res.append(Item(arg_name, arg_desc))
        return res

    def _get_section(self, section: str):
        section_doc = self._sections.get(section, None)
        if section_doc is None:
            res = ''
        else:
            res = _Rd2txt(section_doc)
        return res

    def description(self) -> str:
        """ Get the description of the entry """
        return self._get_section('\\description')

    def details(self) -> str:
        """ Get the section Details for the documentation entry."""
        return self._get_section('\\details')

    def title(self) -> str:
        """ Get the title """
        return self._get_section('\\title')

    def value(self) -> str:
        """ Get the value returned """
        return self._get_section('\\value')

    def seealso(self) -> str:
        """ Get the other documentation entries recommended """
        return self._get_section('\\seealso')

    def usage(self) -> str:
        """ Get the usage for the object """
        return self._get_section('\\usage')

    def items(self):
        """ iterator through the sections names and content
        in the documentation Page. """
        return self.sections.items()

    def iteritems(self):
        """ iterator through the sections names and content
        in the documentation Page. (deprecated, use items()) """
        warnings.warn('Use the method items().', DeprecationWarning)
        return self.sections.items()

    def to_docstring(self, section_names: typing.Optional[typing.Tuple[str, ...]]=None) -> str:
        """ section_names: list of section names to consider. If None
        all sections are used.

        Returns a string that can be used as a Python docstring. """
        s = []
        if section_names is None:
            section_names = self.sections.keys()

        def walk(tree):
            if not isinstance(tree, str):
                for elt in tree:
                    walk(elt)
            else:
                s.append(tree)
                s.append(' ')
        for name in section_names:
            name_str = name[1:] if name.startswith('\\') else name
            s.append(name_str)
            s.append(os.linesep)
            s.append('-' * len(name_str))
            s.append(os.linesep)
            s.append(os.linesep)
            walk(self.sections[name])
            s.append(os.linesep)
            s.append(os.linesep)
        return ''.join(s)