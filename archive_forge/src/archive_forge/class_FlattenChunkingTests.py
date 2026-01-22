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
class FlattenChunkingTests(SynchronousTestCase):
    """
    Tests for the way pieces of the result are chunked together in calls to
    the write function.
    """

    def test_oneSmallChunk(self) -> None:
        """
        If the entire value to be flattened is available synchronously and fits
        into the buffer it is all passed to a single call to the write
        function.
        """
        output: List[bytes] = []
        self.successResultOf(flatten(None, ['1', '2', '3'], output.append))
        assert_that(output, equal_to([b'123']))

    def test_someLargeChunks(self) -> None:
        """
        If the entire value to be flattened is available synchronously but does
        not fit into the buffer then it is chunked into buffer-sized pieces
        and these are passed to the write function.
        """
        some = ['x'] * BUFFER_SIZE
        someMore = ['y'] * BUFFER_SIZE
        evenMore = ['z'] * BUFFER_SIZE
        output: List[bytes] = []
        self.successResultOf(flatten(None, [some, someMore, evenMore], output.append))
        assert_that(output, equal_to([b'x' * BUFFER_SIZE, b'y' * BUFFER_SIZE, b'z' * BUFFER_SIZE]))

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

    def test_chunksSeparatedByFiredDeferred(self) -> None:
        """
        When a fired L{Deferred} is encountered any buffered data is
        passed to the write function.  Then the L{Deferred}'s result is passed
        to another write along with following synchronous values.

        This exact buffering behavior should be considered an implementation
        detail and can be replaced by some other better behavior in the future
        if someone wants.
        """

        def sync_start(v: Flattenable) -> Tuple[Deferred[Flattenable], Callable[[], None]]:
            return (succeed(v), lambda: None)
        self._chunksSeparatedByAsyncTest(sync_start)

    def test_chunksSeparatedByUnfiredDeferred(self) -> None:
        """
        When an unfired L{Deferred} is encountered any buffered data is
        passed to the write function.  After the result of the L{Deferred} is
        available it is passed to another write along with following
        synchronous values.
        """

        def async_start(v: Flattenable) -> Tuple[Deferred[Flattenable], Callable[[], None]]:
            d: Deferred[Flattenable] = Deferred()
            return (d, lambda: d.callback(v))
        self._chunksSeparatedByAsyncTest(async_start)