import aifc
import audioop
import struct
import sunau
import wave
from .exceptions import DecodeError
from .base import AudioFile
Generates blocks of PCM data found in the file.