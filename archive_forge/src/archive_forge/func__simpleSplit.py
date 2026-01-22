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
def _simpleSplit(txt, mW, SW):
    L = []
    ws = SW(' ')
    O = []
    w = -ws
    for t in txt.split():
        lt = SW(t)
        if w + ws + lt <= mW or O == []:
            O.append(t)
            w = w + ws + lt
        else:
            L.append(' '.join(O))
            O = [t]
            w = lt
    if O != []:
        L.append(' '.join(O))
    return L