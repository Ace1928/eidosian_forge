from random import Random
from typing import Awaitable, Dict, List, TypeVar, Union
from hamcrest import (
from hypothesis import given
from hypothesis.strategies import binary, integers, just, lists, randoms, text
from twisted.internet.defer import Deferred, fail
from twisted.internet.interfaces import IProtocol
from twisted.internet.protocol import Protocol
from twisted.protocols.amp import AMP
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, connect
from twisted.trial.unittest import SynchronousTestCase
from ..stream import StreamOpen, StreamReceiver, StreamWrite, chunk, stream
from .matchers import HasSum, IsSequenceOf
class StreamReceiverTests(SynchronousTestCase):
    """
    Tests for L{StreamReceiver}
    """

    @given(lists(lists(binary())), randoms())
    def test_streamReceived(self, streams: List[List[bytes]], random: Random) -> None:
        """
        All data passed to L{StreamReceiver.write} is returned by a call to
        L{StreamReceiver.finish} with a matching C{streamId}.
        """
        receiver = StreamReceiver()
        streamIds = [receiver.open() for _ in streams]
        random.shuffle(streamIds)
        expectedData = dict(zip(streamIds, streams))
        for streamId, strings in expectedData.items():
            for s in strings:
                receiver.write(streamId, s)
        random.shuffle(streamIds)
        actualData = {streamId: receiver.finish(streamId) for streamId in streamIds}
        assert_that(actualData, is_(equal_to(expectedData)))

    @given(integers(), just('data'))
    def test_writeBadStreamId(self, streamId: int, data: str) -> None:
        """
        L{StreamReceiver.write} raises L{KeyError} if called with a
        streamId not associated with an open stream.
        """
        receiver = StreamReceiver()
        assert_that(calling(receiver.write).with_args(streamId, data), raises(KeyError))

    @given(integers())
    def test_badFinishStreamId(self, streamId: int) -> None:
        """
        L{StreamReceiver.finish} raises L{KeyError} if called with a
        streamId not associated with an open stream.
        """
        receiver = StreamReceiver()
        assert_that(calling(receiver.finish).with_args(streamId), raises(KeyError))

    def test_finishRemovesStream(self) -> None:
        """
        L{StreamReceiver.finish} removes the identified stream.
        """
        receiver = StreamReceiver()
        streamId = receiver.open()
        receiver.finish(streamId)
        assert_that(calling(receiver.finish).with_args(streamId), raises(KeyError))