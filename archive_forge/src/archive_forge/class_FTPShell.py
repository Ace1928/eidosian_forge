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
class FTPShell(FTPAnonymousShell):
    """
    An authenticated implementation of L{IFTPShell}.
    """

    def makeDirectory(self, path):
        p = self._path(path)
        try:
            p.makedirs()
        except OSError as e:
            return errnoToFailure(e.errno, path)
        except BaseException:
            return defer.fail()
        else:
            return defer.succeed(None)

    def removeDirectory(self, path):
        p = self._path(path)
        if p.isfile():
            return defer.fail(IsNotADirectoryError(path))
        try:
            os.rmdir(p.path)
        except OSError as e:
            return errnoToFailure(e.errno, path)
        except BaseException:
            return defer.fail()
        else:
            return defer.succeed(None)

    def removeFile(self, path):
        p = self._path(path)
        if p.isdir():
            return defer.fail(IsADirectoryError(path))
        try:
            p.remove()
        except OSError as e:
            return errnoToFailure(e.errno, path)
        except BaseException:
            return defer.fail()
        else:
            return defer.succeed(None)

    def rename(self, fromPath, toPath):
        fp = self._path(fromPath)
        tp = self._path(toPath)
        try:
            os.rename(fp.path, tp.path)
        except OSError as e:
            return errnoToFailure(e.errno, fromPath)
        except BaseException:
            return defer.fail()
        else:
            return defer.succeed(None)

    def openForWriting(self, path):
        """
        Open C{path} for writing.

        @param path: The path, as a list of segments, to open.
        @type path: C{list} of C{unicode}
        @return: A L{Deferred} is returned that will fire with an object
            implementing L{IWriteFile} if the file is successfully opened.  If
            C{path} is a directory, or if an exception is raised while trying
            to open the file, the L{Deferred} will fire with an error.
        """
        p = self._path(path)
        if p.isdir():
            return defer.fail(IsADirectoryError(path))
        try:
            fObj = p.open('w')
        except OSError as e:
            return errnoToFailure(e.errno, path)
        except BaseException:
            return defer.fail()
        return defer.succeed(_FileWriter(fObj))