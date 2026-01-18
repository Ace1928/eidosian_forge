from __future__ import absolute_import
import threading
import warnings
import subprocess
import sys
from unittest import SkipTest, TestCase
import twisted
from twisted.python.log import PythonLoggingObserver
from twisted.python import log
from twisted.python.runtime import platform
from twisted.internet.task import Clock
from .._eventloop import EventLoop, ThreadLogObserver, _store
from ..tests import crochet_directory
import sys
import crochet
import sys
from logging import StreamHandler, Formatter, getLogger, DEBUG
import crochet
from twisted.python import log
from twisted.logger import Logger
import time
def test_second_does_nothing(self):
    """
        The second call to setup() does nothing.
        """
    reactor = FakeReactor()
    s = EventLoop(lambda: reactor, lambda f, *g: None)
    s.setup()
    s.setup()
    reactor.started.wait(5)
    self.assertEqual(reactor.runs, 1)