import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
class NTLMMessage(metaclass=_NTLMMessageMeta):
    """Base NTLM message class that defines the pack and unpack functions."""
    MESSAGE_TYPE = 0
    MINIMUM_LENGTH = 0

    def __init__(self, encoding: typing.Optional[str]=None, _b_data: typing.Optional[bytes]=None) -> None:
        self.signature = b'NTLMSSP\x00' + struct.pack('<I', self.MESSAGE_TYPE)
        self._encoding = encoding or 'windows-1252'
        if _b_data:
            if len(_b_data) < self.MINIMUM_LENGTH:
                raise ValueError('Invalid NTLM %s raw byte length' % self.__class__.__name__)
            self._data = memoryview(bytearray(_b_data))
        else:
            self._data = memoryview(b'')

    def pack(self) -> bytes:
        """Packs the structure to bytes."""
        return self._data.tobytes()

    @staticmethod
    def unpack(b_data: bytes, encoding: typing.Optional[str]=None) -> 'NTLMMessage':
        """Unpacks the structure from bytes."""
        return NTLMMessage(encoding=encoding, _b_data=b_data)