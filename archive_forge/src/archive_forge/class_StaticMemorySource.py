import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
class StaticMemorySource(StaticSource):
    """
    Helper class for default implementation of :class:`.StaticSource`.

    Do not use directly. This class is used internally by pyglet.

    Args:
        data (readable buffer): The audio data.
        audio_format (AudioFormat): The audio format.
    """

    def __init__(self, data, audio_format: AudioFormat) -> None:
        """Construct a memory source over the given data buffer."""
        self._file = io.BytesIO(data)
        self._max_offset = len(data)
        self.audio_format = audio_format
        self._duration = len(data) / float(audio_format.bytes_per_second)

    def is_precise(self) -> bool:
        return True

    def seek(self, timestamp: float) -> None:
        """Seek to given timestamp.

        Args:
            timestamp (float): Time where to seek in the source.
        """
        offset = int(timestamp * self.audio_format.bytes_per_second)
        self._file.seek(self.audio_format.align(offset))

    def get_audio_data(self, num_bytes: float, compensation_time: float=0.0) -> Optional[AudioData]:
        """Get next packet of audio data.

        Args:
            num_bytes (int): Maximum number of bytes of data to return.

        Returns:
            :class:`.AudioData`: Next packet of audio data, or ``None`` if
            there is no (more) data.
        """
        offset_before = self._file.tell()
        data = self._file.read(num_bytes)
        if not data:
            return None
        timestamp = float(offset_before) / self.audio_format.bytes_per_second
        duration = len(data) / self.audio_format.bytes_per_second
        return AudioData(data, len(data), timestamp, duration)