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
def title(self) -> str:
    """ Get the title """
    return self._get_section('\\title')