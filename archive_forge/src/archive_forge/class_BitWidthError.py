import aifc
import audioop
import struct
import sunau
import wave
from .exceptions import DecodeError
from .base import AudioFile
class BitWidthError(DecodeError):
    """The file uses an unsupported bit width."""