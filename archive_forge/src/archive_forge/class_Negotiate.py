import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
class Negotiate(NTLMMessage):
    """NTLM Negotiate Message

    This structure represents an NTLM `NEGOTIATE_MESSAGE`_ that can be serialized and deserialized to and from
    bytes.

    Args:
        flags: The `NegotiateFlags` that the client has negotiated.
        domain_name: The `DomainName` of the client authentication domain.
        workstation: The `Workstation` of the client.
        version: The `Version` of the client.
        encoding: The OEM encoding to use for text fields.
        _b_data: The raw bytes of the message to unpack from.

    .. _NEGOTIATE_MESSAGE:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/b34032e5-3aae-4bc6-84c3-c6d80eadf7f2
    """
    MESSAGE_TYPE = MessageType.negotiate
    MINIMUM_LENGTH = 32

    def __init__(self, flags: int=0, domain_name: typing.Optional[str]=None, workstation: typing.Optional[str]=None, version: typing.Optional['Version']=None, encoding: typing.Optional[str]=None, _b_data: typing.Optional[bytes]=None) -> None:
        super(Negotiate, self).__init__(encoding=encoding, _b_data=_b_data)
        if not _b_data:
            b_payload = bytearray()
            payload_offset = 32
            b_version = b''
            if version:
                flags |= NegotiateFlags.version
                b_version = version.pack()
            payload_offset = _pack_payload(b_version, b_payload, payload_offset)[1]
            b_domain_name = b''
            if domain_name:
                flags |= NegotiateFlags.oem_domain_name_supplied
                b_domain_name = domain_name.encode(self._encoding)
            b_domain_name_fields, payload_offset = _pack_payload(b_domain_name, b_payload, payload_offset)
            b_workstation = b''
            if workstation:
                flags |= NegotiateFlags.oem_workstation_supplied
                b_workstation = workstation.encode(self._encoding)
            b_workstation_fields = _pack_payload(b_workstation, b_payload, payload_offset)[0]
            b_data = bytearray(self.signature)
            b_data.extend(struct.pack('<I', flags))
            b_data.extend(b_domain_name_fields)
            b_data.extend(b_workstation_fields)
            b_data.extend(b_payload)
            self._data = memoryview(b_data)

    @property
    def flags(self) -> int:
        """The negotiate flags for the Negotiate message."""
        return struct.unpack('<I', self._data[12:16].tobytes())[0]

    @flags.setter
    def flags(self, value: int) -> None:
        self._data[12:16] = struct.pack('<I', value)

    @property
    def domain_name(self) -> typing.Optional[str]:
        """The name of the client authentication domain."""
        return to_text(_unpack_payload(self._data, 16), encoding=self._encoding, errors='replace', nonstring='passthru')

    @property
    def workstation(self) -> typing.Optional[str]:
        """The name of the client machine."""
        return to_text(_unpack_payload(self._data, 24), encoding=self._encoding, errors='replace', nonstring='passthru')

    @property
    def version(self) -> typing.Optional['Version']:
        """The client NTLM version."""
        payload_offset = self._payload_offset
        if payload_offset >= 40:
            return Version.unpack(self._data[32:40].tobytes())
        else:
            return None

    @property
    def _payload_offset(self) -> int:
        """Gets the offset of the first payload value."""
        return _get_payload_offset(self._data, [16, 24])

    @staticmethod
    def unpack(b_data: bytes, encoding: typing.Optional[str]=None) -> 'Negotiate':
        msg = NTLMMessage.unpack(b_data, encoding=encoding)
        if not isinstance(msg, Negotiate):
            raise ValueError('Input message was not a NTLM Negotiate message')
        return msg