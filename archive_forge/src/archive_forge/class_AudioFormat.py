import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
class AudioFormat:
    """Audio details.

    An instance of this class is provided by sources with audio tracks.  You
    should not modify the fields, as they are used internally to describe the
    format of data provided by the source.

    Args:
        channels (int): The number of channels: 1 for mono or 2 for stereo
            (pyglet does not yet support surround-sound sources).
        sample_size (int): Bits per sample; only 8 or 16 are supported.
        sample_rate (int): Samples per second (in Hertz).
    """

    def __init__(self, channels: int, sample_size: int, sample_rate: int) -> None:
        self.channels = channels
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.bytes_per_frame = sample_size // 8 * channels
        self.bytes_per_second = self.bytes_per_frame * sample_rate
        self.bytes_per_sample = self.bytes_per_frame
        'This attribute is kept for compatibility and should not be used due\n        to a terminology error.\n        This value contains the bytes per audio frame, and using\n        `bytes_per_frame` should be preferred.\n        For the actual amount of bytes per sample, divide `sample_size` by\n        eight.\n        '

    def align(self, num_bytes: int) -> int:
        """Align a given amount of bytes to the audio frame size of this
        audio format, downwards.
        """
        return num_bytes - num_bytes % self.bytes_per_frame

    def align_ceil(self, num_bytes: int) -> int:
        """Align a given amount of bytes to the audio frame size of this
        audio format, upwards.
        """
        return num_bytes + -num_bytes % self.bytes_per_frame

    def timestamp_to_bytes_aligned(self, timestamp: float) -> int:
        """Given a timestamp, return the amount of bytes that an emitter with
        this audio format would have to have played to reach it, aligned
        to the audio frame size.
        """
        return self.align(int(timestamp * self.bytes_per_second))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AudioFormat):
            return self.channels == other.channels and self.sample_size == other.sample_size and (self.sample_rate == other.sample_rate)
        return NotImplemented

    def __repr__(self) -> str:
        return '%s(channels=%d, sample_size=%d, sample_rate=%d)' % (self.__class__.__name__, self.channels, self.sample_size, self.sample_rate)