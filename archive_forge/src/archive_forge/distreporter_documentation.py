from types import TracebackType
from typing import Optional, Tuple, Union
from zope.interface import implementer
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from ..itrial import IReporter, ITestCase

        Queue stopping the test, then unroll the queue.
        