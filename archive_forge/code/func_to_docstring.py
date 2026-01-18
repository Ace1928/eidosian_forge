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