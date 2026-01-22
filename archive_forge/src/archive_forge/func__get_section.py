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
def _get_section(self, section: str):
    section_doc = self._sections.get(section, None)
    if section_doc is None:
        res = ''
    else:
        res = _Rd2txt(section_doc)
    return res