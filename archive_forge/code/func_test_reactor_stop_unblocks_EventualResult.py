from __future__ import absolute_import
import threading
import subprocess
import time
import gc
import sys
import weakref
import tempfile
import os
import inspect
from unittest import SkipTest
from twisted.trial.unittest import TestCase
from twisted.internet.defer import succeed, Deferred, fail, CancelledError
from twisted.python.failure import Failure
from twisted.python import threadable
from twisted.python.runtime import platform
from .._eventloop import (
from .test_setup import FakeReactor
from .. import (
from ..tests import crochet_directory
import os, threading, signal, time, sys
import crochet
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred, CancelledError
import crochet
from crochet import EventualResult
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
import crochet
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
def test_reactor_stop_unblocks_EventualResult(self):
    """
        Any EventualResult.wait() calls still waiting when the reactor has
        stopped will get a ReactorStopped exception.
        """
    program = 'import os, threading, signal, time, sys\n\nfrom twisted.internet.defer import Deferred\nfrom twisted.internet import reactor\n\nimport crochet\ncrochet.setup()\n\n@crochet.run_in_reactor\ndef run():\n    reactor.callLater(0.1, reactor.stop)\n    return Deferred()\n\ner = run()\ntry:\n    er.wait(timeout=10)\nexcept crochet.ReactorStopped:\n    sys.exit(23)\n'
    process = subprocess.Popen([sys.executable, '-c', program], cwd=crochet_directory)
    self.assertEqual(process.wait(), 23)