from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _extract_one(self, tarinfo, path, set_attrs, numeric_owner):
    """Extract from filtered tarinfo to disk"""
    self._check('r')
    try:
        self._extract_member(tarinfo, os.path.join(path, tarinfo.name), set_attrs=set_attrs, numeric_owner=numeric_owner)
    except OSError as e:
        self._handle_fatal_error(e)
    except ExtractError as e:
        self._handle_nonfatal_error(e)