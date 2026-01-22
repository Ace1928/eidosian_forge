from logging import error
import io
import os
from pprint import pformat
import sys
from warnings import warn
from traitlets.utils.importstring import import_item
from IPython.core import magic_arguments, page
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, magic_escapes
from IPython.utils.text import format_screen, dedent, indent
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.ipstruct import Struct
class MagicsDisplay(object):

    def __init__(self, magics_manager, ignore=None):
        self.ignore = ignore if ignore else []
        self.magics_manager = magics_manager

    def _lsmagic(self):
        """The main implementation of the %lsmagic"""
        mesc = magic_escapes['line']
        cesc = magic_escapes['cell']
        mman = self.magics_manager
        magics = mman.lsmagic()
        out = ['Available line magics:', mesc + ('  ' + mesc).join(sorted([m for m, v in magics['line'].items() if v not in self.ignore])), '', 'Available cell magics:', cesc + ('  ' + cesc).join(sorted([m for m, v in magics['cell'].items() if v not in self.ignore])), '', mman.auto_status()]
        return '\n'.join(out)

    def _repr_pretty_(self, p, cycle):
        p.text(self._lsmagic())

    def __str__(self):
        return self._lsmagic()

    def _jsonable(self):
        """turn magics dict into jsonable dict of the same structure

        replaces object instances with their class names as strings
        """
        magic_dict = {}
        mman = self.magics_manager
        magics = mman.lsmagic()
        for key, subdict in magics.items():
            d = {}
            magic_dict[key] = d
            for name, obj in subdict.items():
                try:
                    classname = obj.__self__.__class__.__name__
                except AttributeError:
                    classname = 'Other'
                d[name] = classname
        return magic_dict

    def _repr_json_(self):
        return self._jsonable()