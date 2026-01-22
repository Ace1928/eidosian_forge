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
def _stat_group(self, fp):
    """
        Get the filepath's owner's group.  If this is not implemented
        (say in Windows) return the string "0" since stat-ing a file in
        Windows seems to return C{st_gid=0}.

        (Reference:
        U{http://stackoverflow.com/questions/5275731/os-stat-on-windows})

        @param fp: L{twisted.python.filepath.FilePath}
        @return: C{str} representing the owner's group
        """
    try:
        groupID = fp.getGroupID()
    except NotImplementedError:
        return '0'
    else:
        if grp is not None:
            try:
                return grp.getgrgid(groupID)[0]
            except KeyError:
                pass
        return str(groupID)