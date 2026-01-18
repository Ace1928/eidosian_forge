import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
def timestamp_to_bytes_aligned(self, timestamp: float) -> int:
    """Given a timestamp, return the amount of bytes that an emitter with
        this audio format would have to have played to reach it, aligned
        to the audio frame size.
        """
    return self.align(int(timestamp * self.bytes_per_second))