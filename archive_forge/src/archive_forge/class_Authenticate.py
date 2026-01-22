import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
class Authenticate(NTLMMessage):
    """NTLM Authentication Message

    This structure represents an NTLM `AUTHENTICATION_MESSAGE`_ that can be serialized and deserialized to and from
    bytes.

    Args:
        flags: The `NegotiateFlags` that the client has negotiated.
        lm_challenge_response: The `LmChallengeResponse` for the client's secret.
        nt_challenge_response: The `NtChallengeResponse` for the client's secret.
        domain_name: The `DomainName` for the client.
        username: The `UserName` for the cleint.
        workstation: The `Workstation` for the client.
        encrypted_session_key: The `EncryptedRandomSessionKey` for the set up context.
        version: The `Version` of the client.
        mic: The `MIC` for the authentication exchange.
        encoding: The OEM encoding to use for text fields if `NTLMSSP_NEGOTIATE_UNICODE` was not supported.
        _b_data: The raw NTLM Authenticate bytes to unpack from.

    .. _AUTHENTICATION_MESSAGE:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/033d32cc-88f9-4483-9bf2-b273055038ce
    """
    MESSAGE_TYPE = MessageType.authenticate
    MINIMUM_LENGTH = 64

    def __init__(self, flags: int=0, lm_challenge_response: typing.Optional[bytes]=None, nt_challenge_response: typing.Optional[bytes]=None, domain_name: typing.Optional[str]=None, username: typing.Optional[str]=None, workstation: typing.Optional[str]=None, encrypted_session_key: typing.Optional[bytes]=None, version: typing.Optional['Version']=None, mic: typing.Optional[bytes]=None, encoding: typing.Optional[str]=None, _b_data: typing.Optional[bytes]=None) -> None:
        super(Authenticate, self).__init__(encoding=encoding, _b_data=_b_data)
        if _b_data:
            self._encoding = 'utf-16-le' if self.flags & NegotiateFlags.unicode else self._encoding
        else:
            self._encoding = 'utf-16-le' if flags & NegotiateFlags.unicode else self._encoding
            b_payload = bytearray()
            payload_offset = 64
            b_version = b''
            if version:
                flags |= NegotiateFlags.version
                b_version = version.pack()
            payload_offset = _pack_payload(b_version, b_payload, payload_offset)[1]
            payload_offset = _pack_payload(b'\x00' * 16, b_payload, payload_offset)[1]
            b_lm_response_fields, payload_offset = _pack_payload(lm_challenge_response, b_payload, payload_offset)
            b_nt_response_fields, payload_offset = _pack_payload(nt_challenge_response, b_payload, payload_offset)
            b_domain_fields, payload_offset = _pack_payload(domain_name, b_payload, payload_offset, lambda d: d.encode(self._encoding))
            b_username_fields, payload_offset = _pack_payload(username, b_payload, payload_offset, lambda d: d.encode(self._encoding))
            b_workstation_fields, payload_offset = _pack_payload(workstation, b_payload, payload_offset, lambda d: d.encode(self._encoding))
            if encrypted_session_key:
                flags |= NegotiateFlags.key_exch
            b_session_key_fields = _pack_payload(encrypted_session_key, b_payload, payload_offset)[0]
            b_data = bytearray(self.signature)
            b_data.extend(b_lm_response_fields)
            b_data.extend(b_nt_response_fields)
            b_data.extend(b_domain_fields)
            b_data.extend(b_username_fields)
            b_data.extend(b_workstation_fields)
            b_data.extend(b_session_key_fields)
            b_data.extend(struct.pack('<I', flags))
            b_data.extend(b_payload)
            self._data = memoryview(b_data)
            if mic:
                self.mic = mic

    @property
    def lm_challenge_response(self) -> typing.Optional[bytes]:
        """The LmChallengeResponse or None if not set."""
        return _unpack_payload(self._data, 12)

    @property
    def nt_challenge_response(self) -> typing.Optional[bytes]:
        """The NtChallengeResponse or None if not set."""
        return _unpack_payload(self._data, 20)

    @property
    def domain_name(self) -> typing.Optional[str]:
        """The domain or computer name hosting the user account."""
        return to_text(_unpack_payload(self._data, 28), encoding=self._encoding, nonstring='passthru')

    @property
    def user_name(self) -> typing.Optional[str]:
        """The name of the user to be authenticated."""
        return to_text(_unpack_payload(self._data, 36), encoding=self._encoding, nonstring='passthru')

    @property
    def workstation(self) -> typing.Optional[str]:
        """The name of the computer to which the user is logged on."""
        return to_text(_unpack_payload(self._data, 44), encoding=self._encoding, nonstring='passthru')

    @property
    def encrypted_random_session_key(self) -> typing.Optional[bytes]:
        """The client's encrypted random session key."""
        return _unpack_payload(self._data, 52)

    @property
    def flags(self) -> int:
        """The negotiate flags supported by the client and server."""
        return struct.unpack('<I', self._data[60:64].tobytes())[0]

    @flags.setter
    def flags(self, value: int) -> None:
        self._data[60:64] = struct.pack('<I', value)

    @property
    def version(self) -> typing.Optional['Version']:
        """The client NTLM version."""
        payload_offset = self._payload_offset
        if payload_offset not in [64, 80] and payload_offset >= 72:
            return Version.unpack(self._data[64:72].tobytes())
        else:
            return None

    @property
    def mic(self) -> typing.Optional[bytes]:
        """The MIC for the Authenticate message."""
        mic_offset = self._get_mic_offset()
        if mic_offset:
            return self._data.tobytes()[mic_offset:mic_offset + 16]
        else:
            return None

    @mic.setter
    def mic(self, value: bytes) -> None:
        if len(value) != 16:
            raise ValueError('NTLM Authenticate MIC must be 16 bytes long')
        mic_offset = self._get_mic_offset()
        if mic_offset:
            self._data[mic_offset:mic_offset + 16] = value
        else:
            raise ValueError('Cannot set MIC on an Authenticate message with no MIC present')

    @property
    def _payload_offset(self) -> int:
        """Gets the offset of the first payload value."""
        return _get_payload_offset(self._data, [12, 20, 28, 36, 44, 52])

    def _get_mic_offset(self) -> int:
        """Gets the offset of the MIC structure if present."""
        payload_offset = self._payload_offset
        if payload_offset >= 88:
            return 72
        elif payload_offset >= 80:
            return 64
        else:
            return 0

    @staticmethod
    def unpack(b_data: bytes, encoding: typing.Optional[str]=None) -> 'Authenticate':
        msg = NTLMMessage.unpack(b_data, encoding=encoding)
        if not isinstance(msg, Authenticate):
            raise ValueError('Input message was not a NTLM Authenticate message')
        return msg