from __future__ import annotations
import errno
import io
import os
import pickle
import stat
import sys
import time
from pprint import pformat
from typing import IO, AnyStr, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from unittest import skipIf
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
from twisted.python import filepath
from twisted.python.filepath import FileMode, OtherAnyStr
from twisted.python.runtime import platform
from twisted.python.win32 import ERROR_DIRECTORY
from twisted.trial.unittest import SynchronousTestCase as TestCase
class TrackingFilePath(filepath.FilePath[AnyStr]):
    """
    A subclass of L{filepath.FilePath} which maintains a list of all other paths
    created by clonePath.

    @ivar trackingList: A list of all paths created by this path via
        C{clonePath} (which also includes paths created by methods like
        C{parent}, C{sibling}, C{child}, etc (and all paths subsequently created
        by those paths, etc).

    @type trackingList: C{list} of L{TrackingFilePath}

    @ivar openedFiles: A list of all file objects opened by this
        L{TrackingFilePath} or any other L{TrackingFilePath} in C{trackingList}.

    @type openedFiles: C{list} of C{file}
    """

    def __init__(self, path: AnyStr, alwaysCreate: bool=False, trackingList: Optional[List[Union[TrackingFilePath[str], TrackingFilePath[bytes]]]]=None) -> None:
        filepath.FilePath.__init__(self, path, alwaysCreate)
        if trackingList is None:
            trackingList = []
        self.trackingList: List[Union[TrackingFilePath[str], TrackingFilePath[bytes]]] = trackingList
        self.openedFiles: List[IO[bytes]] = []

    def open(self, mode: FileMode='r') -> IO[bytes]:
        """
        Override 'open' to track all files opened by this path.
        """
        f = filepath.FilePath.open(self, mode)
        self.openedFiles.append(f)
        return f

    def openedPaths(self) -> List[Union[TrackingFilePath[str], TrackingFilePath[bytes]]]:
        """
        Return a list of all L{TrackingFilePath}s associated with this
        L{TrackingFilePath} that have had their C{open()} method called.
        """
        return [path for path in self.trackingList if path.openedFiles]

    def clonePath(self, path: OtherAnyStr, alwaysCreate: bool=False) -> TrackingFilePath[OtherAnyStr]:
        """
        Override L{filepath.FilePath.clonePath} to give the new path a reference
        to the same tracking list.
        """
        clone = TrackingFilePath(path, trackingList=self.trackingList)
        self.trackingList.append(clone)
        return clone