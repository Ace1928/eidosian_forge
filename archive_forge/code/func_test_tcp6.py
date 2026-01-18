from typing import Type, Union
from twisted.internet.endpoints import (
from twisted.internet.testing import MemoryReactor
from twisted.trial.unittest import SynchronousTestCase as TestCase
from .._parser import unparseEndpoint
from .._wrapper import HAProxyWrappingFactory
def test_tcp6(self) -> None:
    """
        Test if the parser generates a wrapped TCP6 endpoint.
        """
    self.onePrefix('haproxy:tcp6:8080', TCP6ServerEndpoint)