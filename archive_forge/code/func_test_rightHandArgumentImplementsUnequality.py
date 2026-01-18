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
def test_rightHandArgumentImplementsUnequality(self):
    """
        The right-hand argument to the non-equality operator is given a
        chance to determine the result of the operation if it is of a type
        unrelated to the L{FancyEqMixin}-based instance on the left-hand
        side.
        """
    self.assertFalse(Record(1, 2) != EqualToEverything())
    self.assertTrue(Record(1, 2) != EqualToNothing())