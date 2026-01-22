import sys
import os
import re
import warnings
from .errors import (
from .spawn import spawn
from .file_util import move_file
from .dir_util import mkpath
from ._modified import newer_group
from .util import split_quoted, execute
from ._log import log
def _check_macro_definitions(self, definitions):
    """Ensures that every element of 'definitions' is a valid macro
        definition, ie. either (name,value) 2-tuple or a (name,) tuple.  Do
        nothing if all definitions are OK, raise TypeError otherwise.
        """
    for defn in definitions:
        if not (isinstance(defn, tuple) and (len(defn) in (1, 2) and (isinstance(defn[1], str) or defn[1] is None)) and isinstance(defn[0], str)):
            raise TypeError("invalid macro definition '%s': " % defn + 'must be tuple (string,), (string, string), or ' + '(string, None)')