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
def _add_check(check, kind, codes, args):
    if check in _checks[kind]:
        _checks[kind][check][0].extend(codes or [])
    else:
        _checks[kind][check] = (codes or [''], args)