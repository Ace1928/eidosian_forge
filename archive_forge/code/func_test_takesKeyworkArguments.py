import errno
import os.path
import shutil
import sys
import warnings
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.python import util
from twisted.python.filepath import FilePath
from twisted.test.test_process import MockOS
from twisted.trial.unittest import FailTest, TestCase
from twisted.trial.util import suppress as SUPPRESS
def test_takesKeyworkArguments(self):
    """
        L{util.runAsEffectiveUser} pass the keyword parameters to the given
        function.
        """
    result = util.runAsEffectiveUser(0, 0, lambda x, y=1, z=1: x * y * z, 2, z=3)
    self.assertEqual(result, 6)