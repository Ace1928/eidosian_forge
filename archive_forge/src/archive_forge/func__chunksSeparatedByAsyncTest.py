import re
import sys
import traceback
from collections import OrderedDict
from textwrap import dedent
from types import FunctionType
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, cast
from xml.etree.ElementTree import XML
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.defer import (
from twisted.python.failure import Failure
from twisted.test.testutils import XMLAssertionMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.web._flatten import BUFFER_SIZE
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
from twisted.web.template import (
from twisted.web.test._util import FlattenTestCase
def _chunksSeparatedByAsyncTest(self, start: Callable[[Flattenable], Tuple[Deferred[Flattenable], Callable[[], object]]]) -> None:
    """
        Assert that flattening with a L{Deferred} returned by C{start} results
        in the expected buffering behavior.

        The L{Deferred} need not have a result by it is returned by C{start}
        but must have a result after the callable returned along with it is
        called.

        The expected buffering behavior is that flattened values up to the
        L{Deferred} are written together and then the result of the
        L{Deferred} is written together with values following it up to the
        next L{Deferred}.
        """
    first_wait, first_finish = start('first-')
    second_wait, second_finish = start('second-')
    value = ['already-available', '-chunks', first_wait, 'chunks-already-', 'computed', second_wait, 'more-chunks-', 'already-available']
    output: List[bytes] = []
    d = flatten(None, value, output.append)
    first_finish()
    second_finish()
    self.successResultOf(d)
    assert_that(output, equal_to([b'already-available-chunks', b'first-chunks-already-computed', b'second-more-chunks-already-available']))