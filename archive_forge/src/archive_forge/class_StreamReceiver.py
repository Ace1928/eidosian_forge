from itertools import count
from typing import Dict, Iterator, List, TypeVar
from attrs import Factory, define
from twisted.protocols.amp import AMP, Command, Integer, String as Bytes
@define
class StreamReceiver:
    """
    Buffering de-multiplexing byte stream receiver.
    """
    _counter: Iterator[int] = count()
    _streams: Dict[int, List[bytes]] = Factory(dict)

    def open(self) -> int:
        """
        Open a new stream and return its unique identifier.
        """
        newId = next(self._counter)
        self._streams[newId] = []
        return newId

    def write(self, streamId: int, chunk: bytes) -> None:
        """
        Write to an open stream using its unique identifier.

        @raise KeyError: If there is no such open stream.
        """
        self._streams[streamId].append(chunk)

    def finish(self, streamId: int) -> List[bytes]:
        """
        Indicate an open stream may receive no further data and return all of
        its current contents.

        @raise KeyError: If there is no such open stream.
        """
        return self._streams.pop(streamId)