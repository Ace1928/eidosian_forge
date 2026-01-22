import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
class AudioData:
    """A single packet of audio data.

    This class is used internally by pyglet.

    Args:
        data (bytes, ctypes array, or supporting buffer protocol): Sample data.
        length (int): Size of sample data, in bytes.
        timestamp (float): Time of the first sample, in seconds.
        duration (float): Total data duration, in seconds.
        events (List[:class:`pyglet.media.drivers.base.MediaEvent`]): List of events
            contained within this packet. Events are timestamped relative to
            this audio packet.

    .. deprecated:: 2.0.10
            `timestamp` and `duration` are unused and will be removed eventually.
    """
    __slots__ = ('data', 'length', 'timestamp', 'duration', 'events', 'pointer')

    def __init__(self, data: Union[bytes, ctypes.Array], length: int, timestamp: float=0.0, duration: float=0.0, events: Optional[List['MediaEvent']]=None) -> None:
        if isinstance(data, bytes):
            self.pointer = ctypes.cast(data, ctypes.c_void_p).value
        elif isinstance(data, ctypes.Array):
            self.pointer = ctypes.addressof(data)
        else:
            try:
                self.pointer = ctypes.addressof(ctypes.c_int.from_buffer(data))
            except TypeError:
                raise TypeError('Unsupported AudioData type.')
        self.data = data
        self.length = length
        self.timestamp = timestamp
        self.duration = duration
        self.events = [] if events is None else events