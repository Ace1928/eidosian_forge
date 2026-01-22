import argparse
import base64
import json
import os.path
import re
import struct
import sys
import typing
from spnego._asn1 import (
from spnego._context import GSSMech
from spnego._kerberos import (
from spnego._ntlm_raw.crypto import hmac_md5, ntowfv1, ntowfv2, rc4k
from spnego._ntlm_raw.messages import (
from spnego._spnego import InitialContextToken, NegTokenInit, NegTokenResp, unpack_token
from spnego._text import to_bytes
from spnego._tls_struct import (
def _parse_tls_handshake_server_hello(view: memoryview) -> typing.Dict[str, typing.Any]:
    protocol_version = TlsProtocolVersion(struct.unpack('>H', view[:2])[0])
    view = view[2:]
    random = view[:32]
    view = view[32:]
    session_id_len = struct.unpack('B', view[:1])[0]
    view = view[1:]
    session_id = view[:session_id_len]
    view = view[session_id_len:]
    cipher_suite = TlsCipherSuite(struct.unpack('>H', view[:2])[0])
    view = view[2:]
    compression_method = TlsCompressionMethod(struct.unpack('B', view[:1])[0])
    view = view[1:]
    extensions_len = struct.unpack('>H', view[:2])[0]
    view = view[2:]
    extensions_view = view[:extensions_len]
    view = view[extensions_len:]
    extensions = _parse_tls_extensions(extensions_view, False)
    return {'ProtocolVersion': parse_enum(protocol_version), 'Random': base64.b16encode(random).decode(), 'SessionID': base64.b16encode(session_id).decode(), 'CipherSuite': f'{cipher_suite.name} - 0x{cipher_suite.value:04X}', 'CompressionMethod': parse_enum(compression_method), 'Extensions': extensions}