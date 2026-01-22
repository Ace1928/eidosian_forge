import base64
import logging
import os
import socket
import typing
from spnego._context import (
from spnego._credential import (
from spnego._ntlm_raw.crypto import (
from spnego._ntlm_raw.messages import (
from spnego._ntlm_raw.security import seal, sign
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.iov import BufferType, IOVResBuffer
def _compute_response(self, challenge: Challenge, credential: _NTLMCredential, channel_bindings: typing.Optional[GssChannelBindings]) -> typing.Tuple[bytes, bytes, bytes]:
    """Compute the NT and LM responses and the key exchange key."""
    client_challenge = os.urandom(8)
    if self._nt_v2:
        target_info = challenge.target_info.copy() if challenge.target_info else TargetInfo()
        if AvId.timestamp in target_info:
            time = target_info[AvId.timestamp]
            self._mic_required = True
        else:
            time = FileTime.now()
        if channel_bindings:
            target_info[AvId.channel_bindings] = md5(channel_bindings.pack())
        target_info[AvId.target_name] = self.spn or ''
        if self._mic_required:
            target_info[AvId.flags] = target_info.get(AvId.flags, AvFlags(0)) | AvFlags.mic
        ntv2_hash = ntowfv2(credential.username or '', credential.nt_hash, credential.domain)
        nt_challenge, lm_challenge, key_exchange_key = compute_response_v2(ntv2_hash, challenge.server_challenge, client_challenge, time, target_info)
        if self._mic_required:
            lm_challenge = b'\x00' * 24
        return (nt_challenge, lm_challenge, key_exchange_key)
    else:
        return compute_response_v1(challenge.flags, credential.nt_hash, credential.lm_hash, challenge.server_challenge, client_challenge, no_lm_response=not self._lm)