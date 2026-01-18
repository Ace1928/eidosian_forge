import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def url_from_path(path):
    fspath = path_to_fspath(path, False)
    from urllib import quote
    if ISWINDOWS:
        match = _reg_allow_disk.match(fspath)
        fspath = fspath.replace('\\', '/')
        if match.group(1):
            fspath = '/%s%s' % (match.group(1).replace('\\', '/'), quote(fspath[len(match.group(1)):]))
        else:
            fspath = quote(fspath)
    else:
        fspath = quote(fspath)
    if path.rev != -1:
        fspath = '%s@%s' % (fspath, path.rev)
    else:
        fspath = '%s@HEAD' % (fspath,)
    return 'file://%s' % (fspath,)