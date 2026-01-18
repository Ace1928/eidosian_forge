import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
def write_section(name: str, section_dict: _OMD) -> None:
    fp.write(('[%s]\n' % name).encode(defenc))
    values: Sequence[str]
    v: str
    for key, values in section_dict.items_all():
        if key == '__name__':
            continue
        for v in values:
            fp.write(('\t%s = %s\n' % (key, self._value_to_string(v).replace('\n', '\n\t'))).encode(defenc))