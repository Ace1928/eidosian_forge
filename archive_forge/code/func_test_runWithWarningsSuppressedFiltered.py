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
def test_runWithWarningsSuppressedFiltered(self):
    """
        Warnings from the function called by C{runWithWarningsSuppressed} are
        suppressed if they match the passed in filter.
        """
    filters = [(('ignore', '.*foo.*'), {}), (('ignore', '.*bar.*'), {})]
    self.runWithWarningsSuppressed(filters, warnings.warn, 'ignore foo')
    self.runWithWarningsSuppressed(filters, warnings.warn, 'ignore bar')
    self.assertEqual([], self.flushWarnings())