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
def _parse_ntlm_authenticate(data: Authenticate, password: typing.Optional[str]) -> typing.Dict[str, typing.Any]:
    b_data = data.pack()
    msg: typing.Dict[str, typing.Any] = {'LmChallengeResponseFields': {'Len': struct.unpack('<H', b_data[12:14])[0], 'MaxLen': struct.unpack('<H', b_data[14:16])[0], 'BufferOffset': struct.unpack('<I', b_data[16:20])[0]}, 'NtChallengeResponseFields': {'Len': struct.unpack('<H', b_data[20:22])[0], 'MaxLen': struct.unpack('<H', b_data[22:24])[0], 'BufferOffset': struct.unpack('<I', b_data[24:28])[0]}, 'DomainNameFields': {'Len': struct.unpack('<H', b_data[28:30])[0], 'MaxLen': struct.unpack('<H', b_data[30:32])[0], 'BufferOffset': struct.unpack('<I', b_data[32:36])[0]}, 'UserNameFields': {'Len': struct.unpack('<H', b_data[36:38])[0], 'MaxLen': struct.unpack('<H', b_data[38:40])[0], 'BufferOffset': struct.unpack('<I', b_data[40:44])[0]}, 'WorkstationFields': {'Len': struct.unpack('<H', b_data[44:46])[0], 'MaxLen': struct.unpack('<H', b_data[46:48])[0], 'BufferOffset': struct.unpack('<I', b_data[48:52])[0]}, 'EncryptedRandomSessionKeyFields': {'Len': struct.unpack('<H', b_data[52:54])[0], 'MaxLen': struct.unpack('<H', b_data[54:56])[0], 'BufferOffset': struct.unpack('<I', b_data[56:60])[0]}, 'NegotiateFlags': parse_flags(data.flags, enum_type=NegotiateFlags), 'Version': _parse_ntlm_version(data.version), 'MIC': base64.b16encode(data.mic).decode() if data.mic else None, 'Payload': {'LmChallengeResponse': None, 'NtChallengeResponse': None, 'DomainName': data.domain_name, 'UserName': data.user_name, 'Workstation': data.workstation, 'EncryptedRandomSessionKey': None}}
    key_exchange_key = None
    lm_response_data = data.lm_challenge_response
    nt_response_data = data.nt_challenge_response
    if lm_response_data:
        lm_response: typing.Dict[str, typing.Any] = {'ResponseType': None, 'LMProofStr': None}
        if not nt_response_data or len(nt_response_data) == 24:
            lm_response['ResponseType'] = 'LMv1'
            lm_response['LMProofStr'] = base64.b16encode(lm_response_data).decode()
        else:
            lm_response['ResponseType'] = 'LMv2'
            lm_response['LMProofStr'] = base64.b16encode(lm_response_data[:16]).decode()
            lm_response['ChallengeFromClient'] = base64.b16encode(lm_response_data[16:]).decode()
        msg['Payload']['LmChallengeResponse'] = lm_response
    if nt_response_data:
        nt_response: typing.Dict[str, typing.Any] = {'ResponseType': None, 'NTProofStr': None}
        if len(nt_response_data) == 24:
            nt_response['ResponseType'] = 'NTLMv1'
            nt_response['NTProofStr'] = base64.b16encode(nt_response_data).decode()
        else:
            nt_proof_str = nt_response_data[:16]
            nt_response['ResponseType'] = 'NTLMv2'
            nt_response['NTProofStr'] = base64.b16encode(nt_proof_str).decode()
            challenge = NTClientChallengeV2.unpack(nt_response_data[16:])
            b_challenge = nt_response_data[16:]
            nt_response['ClientChallenge'] = {'RespType': challenge.resp_type, 'HiRespType': challenge.hi_resp_type, 'Reserved1': struct.unpack('<H', b_challenge[2:4])[0], 'Reserved2': struct.unpack('<I', b_challenge[4:8])[0], 'TimeStamp': str(challenge.time_stamp), 'ChallengeFromClient': base64.b16encode(challenge.challenge_from_client).decode(), 'Reserved3': struct.unpack('<I', b_challenge[24:28])[0], 'AvPairs': _parse_ntlm_target_info(challenge.av_pairs), 'Reserved4': struct.unpack('<I', b_challenge[-4:])[0]}
            if password:
                response_key_nt = ntowfv2(msg['Payload']['UserName'], ntowfv1(password), msg['Payload']['DomainName'])
                key_exchange_key = hmac_md5(response_key_nt, nt_proof_str)
        msg['Payload']['NtChallengeResponse'] = nt_response
    if data.encrypted_random_session_key:
        msg['Payload']['EncryptedRandomSessionKey'] = base64.b16encode(data.encrypted_random_session_key).decode()
    if data.flags & NegotiateFlags.key_exch and (data.flags & NegotiateFlags.sign or data.flags & NegotiateFlags.seal):
        session_key = None
        if key_exchange_key:
            session_key = rc4k(key_exchange_key, typing.cast(bytes, data.encrypted_random_session_key))
    else:
        session_key = key_exchange_key
    msg['SessionKey'] = base64.b16encode(session_key).decode() if session_key else 'Failed to derive'
    return msg