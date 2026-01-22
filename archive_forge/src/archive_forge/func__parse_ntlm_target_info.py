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
def _parse_ntlm_target_info(target_info: typing.Optional[TargetInfo]) -> typing.Optional[typing.List[typing.Dict[str, typing.Any]]]:
    if target_info is None:
        return None
    text_values = [AvId.nb_computer_name, AvId.nb_domain_name, AvId.dns_computer_name, AvId.dns_domain_name, AvId.dns_tree_name, AvId.target_name]
    info = []
    for av_id, raw_value in target_info.items():
        if av_id == AvId.eol:
            value = None
        elif av_id in text_values:
            value = raw_value
        elif av_id == AvId.flags:
            value = parse_flags(raw_value)
        elif av_id == AvId.timestamp:
            value = str(raw_value)
        elif av_id == AvId.single_host:
            value = {'Size': raw_value.size, 'Z4': raw_value.z4, 'CustomData': base64.b16encode(raw_value.custom_data).decode(), 'MachineId': base64.b16encode(raw_value.machine_id).decode()}
        else:
            value = base64.b16encode(raw_value).decode()
        info.append({'AvId': parse_enum(av_id), 'Value': value})
    return info