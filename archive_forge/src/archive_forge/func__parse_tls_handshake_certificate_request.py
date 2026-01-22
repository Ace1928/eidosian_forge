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
def _parse_tls_handshake_certificate_request(view: memoryview) -> typing.Dict[str, typing.Any]:
    cert_types_len = struct.unpack('B', view[:1])[0]
    view = view[1:]
    cert_types_view = view[:cert_types_len]
    view = view[cert_types_len:]
    cert_types = []
    while cert_types_view:
        ct = TlsClientCertificateType(struct.unpack('B', cert_types_view[:1])[0])
        cert_types.append(parse_enum(ct))
        cert_types_view = cert_types_view[1:]
    sig_algos_len = struct.unpack('>H', view[:2])[0]
    view = view[2:]
    sig_algos_view = view[:sig_algos_len]
    view = view[sig_algos_len:]
    sig_algos = []
    while sig_algos_view:
        algo = TlsSignatureScheme(struct.unpack('>H', sig_algos_view[:2])[0])
        sig_algos.append(parse_enum(algo))
        sig_algos_view = sig_algos_view[2:]
    dn_len = struct.unpack('>H', view[:2])[0]
    view = view[2:]
    dn_view = view[:dn_len]
    view = view[dn_len:]
    dns = []
    while dn_view:
        entry_len = struct.unpack('>H', dn_view[:2])[0]
        dn_view = dn_view[2:]
        entry_view = dn_view[:entry_len]
        dn_view = dn_view[entry_len:]
        for dn_entry in unpack_asn1_sequence(entry_view):
            for dn_set in unpack_asn1_sequence(dn_entry):
                dn_data = unpack_asn1(dn_set.b_data)[0]
                dn_oid, dn_str = unpack_asn1_sequence(dn_data)
                oid = DistinguishedNameType(unpack_asn1_object_identifier(dn_oid))
                dns.append({'OID': parse_enum(oid), 'Value': dn_str.b_data.tobytes().decode('utf-8')})
    return {'CertificateTypes': cert_types, 'SignatureAlgorithms': sig_algos, 'CertificateAuthorities': dns}