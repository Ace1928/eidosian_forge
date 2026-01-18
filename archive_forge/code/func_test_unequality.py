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
def test_unequality(self):
    """
        Inequality between instances of a particular L{record} should be
        defined as the negation of equality.
        """
    self.assertFalse(Record(1, 2) != Record(1, 2))
    self.assertTrue(Record(1, 2) != Record(1, 3))
    self.assertTrue(Record(1, 2) != Record(2, 2))
    self.assertTrue(Record(1, 2) != Record(3, 4))