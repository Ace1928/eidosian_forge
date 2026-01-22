import sys, os, pickle
from hashlib import md5
from xml.sax.saxutils import quoteattr
from time import process_time as clock
from reportlab.lib.utils import asBytes, asNative as _asNative
from reportlab.lib.utils import rl_isdir, rl_isfile, rl_listdir, rl_getmtime
def addDirectories(self, dirNames, recur=None):
    for dirName in dirNames:
        self.addDirectory(dirName, recur=recur)