import errno
from functools import wraps
from os import getpid, name as SYSTEM_NAME
from typing import Any, Callable, Optional
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
import twisted.trial.unittest
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
from ...runner import _pidfile
from .._pidfile import (
def test_readDoesntExist(self) -> None:
    """
        L{PIDFile.read} raises L{NoPIDFound} when given a non-existing file
        path.
        """
    pidFile = PIDFile(self.filePath())
    e = self.assertRaises(NoPIDFound, pidFile.read)
    self.assertEqual(str(e), 'PID file does not exist')