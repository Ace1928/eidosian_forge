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
def _parse_spnego_resp(data: NegTokenResp, secret: typing.Optional[str]=None, encoding: typing.Optional[str]=None) -> typing.Dict[str, typing.Any]:
    supported_mech = parse_enum(data.supported_mech, enum_type=GSSMech) if data.supported_mech else None
    response_token = None
    if data.response_token:
        response_token = parse_token(data.response_token, secret=secret, encoding=encoding)
    msg = {'negState': parse_enum(data.neg_state) if data.neg_state is not None else None, 'supportedMech': supported_mech, 'responseToken': response_token, 'mechListMIC': base64.b16encode(data.mech_list_mic).decode() if data.mech_list_mic is not None else None}
    return msg