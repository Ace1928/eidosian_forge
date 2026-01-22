from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations
import sys
import os
import time
import zlib
import warnings
class DummyCrashContainer(object):
    """
    Fakes a database of volatile Crash objects,
    trying to mimic part of it's interface, but
    doesn't actually store anything.

    Normally applications don't need to use this.

    @see: L{CrashDictionary}
    """

    def __init__(self, allowRepeatedKeys=True):
        """
        Fake containers don't store L{Crash} objects, but they implement the
        interface properly.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            Mimics the duplicate filter behavior found in real containers.
        """
        self.__keys = set()
        self.__count = 0
        self.__allowRepeatedKeys = allowRepeatedKeys

    def __contains__(self, crash):
        """
        @type  crash: L{Crash}
        @param crash: Crash object.

        @rtype:  bool
        @return: C{True} if the Crash object is in the container.
        """
        return crash.signature in self.__keys

    def __len__(self):
        """
        @rtype:  int
        @return: Count of L{Crash} elements in the container.
        """
        if self.__allowRepeatedKeys:
            return self.__count
        return len(self.__keys)

    def __bool__(self):
        """
        @rtype:  bool
        @return: C{False} if the container is empty.
        """
        return bool(len(self))

    def add(self, crash):
        """
        Adds a new crash to the container.

        @note:
            When the C{allowRepeatedKeys} parameter of the constructor
            is set to C{False}, duplicated crashes are ignored.

        @see: L{Crash.key}

        @type  crash: L{Crash}
        @param crash: Crash object to add.
        """
        self.__keys.add(crash.signature)
        self.__count += 1

    def get(self, key):
        """
        This method is not supported.
        """
        raise NotImplementedError()

    def has_key(self, key):
        """
        @type  key: L{Crash} signature.
        @param key: Heuristic signature of the crash to get.

        @rtype:  bool
        @return: C{True} if a matching L{Crash} object is in the container.
        """
        return self.__keys.has_key(key)

    def iterkeys(self):
        """
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} object keys.

        @see:     L{get}
        @warning: A B{copy} of each object is returned,
            so any changes made to them will be lost.

            To preserve changes do the following:
                1. Keep a reference to the object.
                2. Delete the object from the set.
                3. Modify the object and add it again.
        """
        return iter(self.__keys)