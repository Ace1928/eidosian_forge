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
def _show_module_versions(self, k, v):
    self._writeln(k[2:])
    K = list(v.keys())
    K.sort()
    for k in K:
        vk = vk0 = v[k]
        if isinstance(vk, tuple):
            vk0 = vk[0]
        try:
            __import__(k)
            m = sys.modules[k]
            d = getattr(m, '__version__', None) == vk0 and 'SAME' or 'DIFFERENT'
        except:
            m = None
            d = '??????unknown??????'
        self._writeln('  %s = %s (%s)' % (k, vk, d))