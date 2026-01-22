import os
import stat
from itertools import filterfalse
from types import GenericAlias
def _sig(st):
    return (stat.S_IFMT(st.st_mode), st.st_size, st.st_mtime)