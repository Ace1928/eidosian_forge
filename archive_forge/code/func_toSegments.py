import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def toSegments(cwd, path):
    """
    Normalize a path, as represented by a list of strings each
    representing one segment of the path.
    """
    if path.startswith('/'):
        segs = []
    else:
        segs = cwd[:]
    for s in path.split('/'):
        if s == '.' or s == '':
            continue
        elif s == '..':
            if segs:
                segs.pop()
            else:
                raise InvalidPath(cwd, path)
        elif '\x00' in s or '/' in s:
            raise InvalidPath(cwd, path)
        else:
            segs.append(s)
    return segs