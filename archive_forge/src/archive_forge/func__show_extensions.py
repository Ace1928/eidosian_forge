import os, pickle, sys, time, types, datetime, importlib
from ast import literal_eval
from base64 import decodebytes as base64_decodebytes, encodebytes as base64_encodebytes
from io import BytesIO
from hashlib import md5
from reportlab.lib.rltempfile import get_rl_tempfile, get_rl_tempdir
from . rl_safe_eval import rl_safe_exec, rl_safe_eval, safer_globals, rl_extended_literal_eval
from PIL import Image
import builtins
import reportlab
import glob, fnmatch
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from importlib import util as importlib_util
import itertools
def _show_extensions(self):
    for mn in ('_rl_accel', '_renderPM', 'sgmlop', 'pyRXP', 'pyRXPU', '_imaging', 'Image'):
        try:
            A = [mn].append
            __import__(mn)
            m = sys.modules[mn]
            A(m.__file__)
            for vn in ('__version__', 'VERSION', '_version', 'version'):
                if hasattr(m, vn):
                    A('%s=%r' % (vn, getattr(m, vn)))
        except:
            A('not found')
        self._writeln(' ' + ' '.join(A.__self__))