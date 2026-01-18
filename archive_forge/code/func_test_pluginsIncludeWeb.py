from sys import stderr, stdout
from typing import Callable, Dict, List, Optional, TextIO, Tuple
import twisted.trial.unittest
from twisted.copyright import version
from twisted.internet import reactor
from twisted.internet.interfaces import IReactorCore
from twisted.internet.testing import MemoryReactor
from twisted.logger import (
from twisted.python.usage import UsageError
from ...reactors import NoSuchReactor
from ...runner._exit import ExitStatus
from ...runner.test.test_runner import DummyExit
from ...service import ServiceMaker
from ...twist import _options
from .._options import TwistOptions
def test_pluginsIncludeWeb(self) -> None:
    """
        L{TwistOptions.plugins} includes a C{"web"} plug-in.
        This is an attempt to verify that something we expect to be in the list
        is in there without enumerating all of the built-in plug-ins.
        """
    options = TwistOptions()
    self.assertIn('web', options.plugins)