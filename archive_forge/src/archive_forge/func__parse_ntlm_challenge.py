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
def _parse_ntlm_challenge(data: Challenge) -> typing.Dict[str, typing.Any]:
    b_data = data.pack()
    msg = {'TargetNameFields': {'Len': struct.unpack('<H', b_data[12:14])[0], 'MaxLen': struct.unpack('<H', b_data[14:16])[0], 'BufferOffset': struct.unpack('<I', b_data[16:20])[0]}, 'NegotiateFlags': parse_flags(data.flags, enum_type=NegotiateFlags), 'ServerChallenge': base64.b16encode(b_data[24:32]).decode(), 'Reserved': base64.b16encode(b_data[32:40]).decode(), 'TargetInfoFields': {'Len': struct.unpack('<H', b_data[40:42])[0], 'MaxLen': struct.unpack('<H', b_data[42:44])[0], 'BufferOffset': struct.unpack('<I', b_data[44:48])[0]}, 'Version': _parse_ntlm_version(data.version), 'Payload': {'TargetName': data.target_name, 'TargetInfo': _parse_ntlm_target_info(data.target_info)}}
    return msg