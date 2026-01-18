import sys
import pickle
import errno
import subprocess as sp
import gzip
import hashlib
import locale
from hashlib import md5
import os
import os.path as op
import re
import shutil
import contextlib
import posixpath
from pathlib import Path
import simplejson as json
from time import sleep, time
from .. import logging, config, __version__ as version
from .misc import is_container
def write_rst_list(items, prefix=''):
    out = []
    for item in ensure_list(items):
        out.append('{} {}'.format(prefix, str(item)))
    return '\n'.join(out) + '\n\n'