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
def _securedFunction(self, startUID, startGID, wantUID, wantGID):
    """
        Check if wanted UID/GID matched start or saved ones.
        """
    self.assertTrue(wantUID == startUID or wantUID == self.mockos.seteuidCalls[-1])
    self.assertTrue(wantGID == startGID or wantGID == self.mockos.setegidCalls[-1])