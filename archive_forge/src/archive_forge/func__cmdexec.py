import os, sys, time, re
import py
from py import path, process
from py._path import common
from py._path import svnwc as svncommon
from py._path.cacheutil import BuildcostAccessCache, AgingCache
def _cmdexec(self, cmd):
    try:
        out = process.cmdexec(cmd)
    except py.process.cmdexec.Error:
        e = sys.exc_info()[1]
        if e.err.find('File Exists') != -1 or e.err.find('File already exists') != -1:
            raise py.error.EEXIST(self)
        raise
    return out