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
def _parse_tls_extensions(view: memoryview, is_client_hello: bool) -> typing.List[typing.Dict[str, typing.Any]]:
    extensions = []
    while view:
        ext_type = TlsExtensionType(struct.unpack('>H', view[:2])[0])
        view = view[2:]
        ext_len = struct.unpack('>H', view[:2])[0]
        view = view[2:]
        ext_data = view[:ext_len]
        view = view[ext_len:]
        data: typing.Any = None
        if ext_len == 0:
            data = None
        elif ext_type == TlsExtensionType.server_name:
            data_len = struct.unpack('>H', ext_data[:2])[0]
            data_view = ext_data[2:2 + data_len]
            data = []
            while data_view:
                name_type = TlsServerNameType(struct.unpack('B', data_view[:1])[0])
                name_len = struct.unpack('>H', data_view[1:3])[0]
                name = data_view[3:3 + name_len].tobytes().decode('utf-8')
                data.append({'Type': parse_enum(name_type), 'Name': name})
                data_view = data_view[3 + name_len:]
        elif ext_type == TlsExtensionType.ec_point_formats:
            data_len = struct.unpack('B', ext_data[:1])[0]
            data_view = ext_data[1:1 + data_len]
            data = [parse_enum(TlsECPointFormat(b)) for b in list(data_view)]
        elif ext_type == TlsExtensionType.supported_groups:
            data_len = struct.unpack('>H', ext_data[:2])[0]
            data_view = ext_data[2:2 + data_len]
            data = []
            while data_view:
                data.append(parse_enum(TlsSupportedGroup(struct.unpack('>H', data_view[:2])[0])))
                data_view = data_view[2:]
        elif ext_type == TlsExtensionType.application_layer_protocol_negotiation:
            data_len = struct.unpack('>H', ext_data[:2])[0]
            data_view = ext_data[2:2 + data_len]
            data = []
            while data_view:
                alpn_len = struct.unpack('B', data_view[:1])[0]
                data_view = data_view[1:]
                data.append(data_view[:alpn_len].tobytes().decode())
                data_view = data_view[alpn_len:]
        elif ext_type == TlsExtensionType.session_ticket:
            data = base64.b16encode(ext_data.tobytes()).decode()
        elif ext_type == TlsExtensionType.signature_algorithms:
            data_len = struct.unpack('>H', ext_data[:2])[0]
            data_view = ext_data[2:2 + data_len]
            data = []
            while data_view:
                data.append(parse_enum(TlsSignatureScheme(struct.unpack('>H', data_view[:2])[0])))
                data_view = data_view[2:]
        elif ext_type == TlsExtensionType.supported_versions:
            if is_client_hello:
                data_len = struct.unpack('B', ext_data[:1])[0]
                data_view = ext_data[1:1 + data_len]
                data = []
                while data_view:
                    data.append(parse_enum(TlsProtocolVersion(struct.unpack('>H', data_view[:2])[0])))
                    data_view = data_view[2:]
            else:
                data = parse_enum(TlsProtocolVersion(struct.unpack('>H', ext_data[:2])[0]))
        elif ext_type == TlsExtensionType.psk_key_exchange_modes:
            data_len = struct.unpack('B', ext_data[:1])[0]
            data_view = ext_data[1:1 + data_len]
            data = []
            while data_view:
                data.append(parse_enum(TlsPskKeyExchangeMode(struct.unpack('B', data_view[:1])[0])))
                data_view = data_view[1:]
        elif ext_type == TlsExtensionType.key_share:
            if is_client_hello:
                data_len = struct.unpack('>H', ext_data[:2])[0]
                data_view = ext_data[2:2 + data_len]
                data = []
                while data_view:
                    key_share_group = TlsSupportedGroup(struct.unpack('>H', data_view[:2])[0])
                    key_exchange_len = struct.unpack('>H', data_view[2:4])[0]
                    key_exchange = data_view[4:4 + key_exchange_len].tobytes()
                    data.append({'Group': parse_enum(key_share_group), 'Key': base64.b16encode(key_exchange).decode()})
                    data_view = data_view[4 + key_exchange_len:]
            else:
                key_share_group = TlsSupportedGroup(struct.unpack('>H', ext_data[:2])[0])
                key_exchange_len = struct.unpack('>H', ext_data[2:4])[0]
                key_exchange = ext_data[4:4 + key_exchange_len].tobytes()
                data = {'Group': parse_enum(key_share_group), 'Key': base64.b16encode(key_exchange).decode()}
        formated_data = {'ExtensionType': parse_enum(ext_type)}
        if data is not None:
            formated_data['Data'] = data
        if data is not None or ext_data:
            formated_data['RawData'] = base64.b16encode(ext_data).decode()
        extensions.append(formated_data)
    return extensions