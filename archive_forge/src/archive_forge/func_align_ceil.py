import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
def align_ceil(self, num_bytes: int) -> int:
    """Align a given amount of bytes to the audio frame size of this
        audio format, upwards.
        """
    return num_bytes + -num_bytes % self.bytes_per_frame