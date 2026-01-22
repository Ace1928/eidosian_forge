from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def _unichr_is_printable(char):
    if char == u' ':
        return True
    import unicodedata
    return unicodedata.category(char) not in ('C', 'Z')