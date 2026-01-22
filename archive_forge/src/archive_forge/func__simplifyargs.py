import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def _simplifyargs(argsline):
    a = []
    for n in markoutercomma(argsline).split('@,@'):
        for r in '(),':
            n = n.replace(r, '_')
        a.append(n)
    return ','.join(a)