from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def _is_eol_token(token, _eol_token=_is_eol_token):
    return _eol_token(token) or (token[0] == tokenize.COMMENT and token[1] == token[4])