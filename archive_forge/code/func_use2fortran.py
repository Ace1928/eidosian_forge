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
def use2fortran(use, tab=''):
    ret = ''
    for m in list(use.keys()):
        ret = '%s%suse %s,' % (ret, tab, m)
        if use[m] == {}:
            if ret and ret[-1] == ',':
                ret = ret[:-1]
            continue
        if 'only' in use[m] and use[m]['only']:
            ret = '%s only:' % ret
        if 'map' in use[m] and use[m]['map']:
            c = ' '
            for k in list(use[m]['map'].keys()):
                if k == use[m]['map'][k]:
                    ret = '%s%s%s' % (ret, c, k)
                    c = ','
                else:
                    ret = '%s%s%s=>%s' % (ret, c, k, use[m]['map'][k])
                    c = ','
        if ret and ret[-1] == ',':
            ret = ret[:-1]
    return ret