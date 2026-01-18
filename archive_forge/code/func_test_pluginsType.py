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
def test_pluginsType(self) -> None:
    """
        L{TwistOptions.plugins} is a mapping of available plug-ins.
        """
    options = TwistOptions()
    plugins = options.plugins
    for name in plugins:
        self.assertIsInstance(name, str)
        self.assertIsInstance(plugins[name], ServiceMaker)