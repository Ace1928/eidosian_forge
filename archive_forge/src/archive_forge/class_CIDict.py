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
class CIDict(dict):

    def __init__(self, *args, **kwds):
        for a in args:
            self.update(a)
        self.update(kwds)

    def update(self, D):
        for k, v in D.items():
            self[k] = v

    def __setitem__(self, k, v):
        try:
            k = k.lower()
        except:
            pass
        dict.__setitem__(self, k, v)

    def __getitem__(self, k):
        try:
            k = k.lower()
        except:
            pass
        return dict.__getitem__(self, k)

    def __delitem__(self, k):
        try:
            k = k.lower()
        except:
            pass
        return dict.__delitem__(self, k)

    def get(self, k, dv=None):
        try:
            return self[k]
        except KeyError:
            return dv

    def __contains__(self, k):
        try:
            self[k]
            return True
        except:
            return False

    def pop(self, k, *a):
        try:
            k = k.lower()
        except:
            pass
        return dict.pop(*(self, k) + a)

    def setdefault(self, k, *a):
        try:
            k = k.lower()
        except:
            pass
        return dict.setdefault(*(self, k) + a)