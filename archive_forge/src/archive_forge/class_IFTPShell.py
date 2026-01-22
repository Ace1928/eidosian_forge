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
class IFTPShell(Interface):
    """
    An abstraction of the shell commands used by the FTP protocol for
    a given user account.

    All path names must be absolute.
    """

    def makeDirectory(path):
        """
        Create a directory.

        @param path: The path, as a list of segments, to create
        @type path: C{list} of C{unicode}

        @return: A Deferred which fires when the directory has been
        created, or which fails if the directory cannot be created.
        """

    def removeDirectory(path):
        """
        Remove a directory.

        @param path: The path, as a list of segments, to remove
        @type path: C{list} of C{unicode}

        @return: A Deferred which fires when the directory has been
        removed, or which fails if the directory cannot be removed.
        """

    def removeFile(path):
        """
        Remove a file.

        @param path: The path, as a list of segments, to remove
        @type path: C{list} of C{unicode}

        @return: A Deferred which fires when the file has been
        removed, or which fails if the file cannot be removed.
        """

    def rename(fromPath, toPath):
        """
        Rename a file or directory.

        @param fromPath: The current name of the path.
        @type fromPath: C{list} of C{unicode}

        @param toPath: The desired new name of the path.
        @type toPath: C{list} of C{unicode}

        @return: A Deferred which fires when the path has been
        renamed, or which fails if the path cannot be renamed.
        """

    def access(path):
        """
        Determine whether access to the given path is allowed.

        @param path: The path, as a list of segments

        @return: A Deferred which fires with None if access is allowed
        or which fails with a specific exception type if access is
        denied.
        """

    def stat(path, keys=()):
        """
        Retrieve information about the given path.

        This is like list, except it will never return results about
        child paths.
        """

    def list(path, keys=()):
        """
        Retrieve information about the given path.

        If the path represents a non-directory, the result list should
        have only one entry with information about that non-directory.
        Otherwise, the result list should have an element for each
        child of the directory.

        @param path: The path, as a list of segments, to list
        @type path: C{list} of C{unicode} or C{bytes}

        @param keys: A tuple of keys desired in the resulting
        dictionaries.

        @return: A Deferred which fires with a list of (name, list),
        where the name is the name of the entry as a unicode string or
        bytes and each list contains values corresponding to the requested
        keys.  The following are possible elements of keys, and the
        values which should be returned for them:

            - C{'size'}: size in bytes, as an integer (this is kinda required)

            - C{'directory'}: boolean indicating the type of this entry

            - C{'permissions'}: a bitvector (see os.stat(foo).st_mode)

            - C{'hardlinks'}: Number of hard links to this entry

            - C{'modified'}: number of seconds since the epoch since entry was
              modified

            - C{'owner'}: string indicating the user owner of this entry

            - C{'group'}: string indicating the group owner of this entry
        """

    def openForReading(path):
        """
        @param path: The path, as a list of segments, to open
        @type path: C{list} of C{unicode}

        @rtype: C{Deferred} which will fire with L{IReadFile}
        """

    def openForWriting(path):
        """
        @param path: The path, as a list of segments, to open
        @type path: C{list} of C{unicode}

        @rtype: C{Deferred} which will fire with L{IWriteFile}
        """