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
def _step_accept_authenticate(self, token: bytes, channel_bindings: typing.Optional[GssChannelBindings]) -> None:
    """Process the Authenticate message from the initiator."""
    negotiate = typing.cast(Negotiate, self._temp_negotiate)
    challenge = typing.cast(Challenge, self._temp_challenge)
    server_challenge = challenge.server_challenge
    auth = Authenticate.unpack(token)
    if not auth.user_name or (not auth.nt_challenge_response and (not auth.lm_challenge_response or auth.lm_challenge_response == b'\x00')):
        raise OperationNotAvailableError(context_msg='Anonymous user authentication not implemented')
    username = auth.user_name
    if auth.domain_name:
        username = f'{auth.domain_name}\\{username}'
    self._credential = _NTLMCredential(CredentialCache(username=username))
    expected_mic = None
    if auth.nt_challenge_response and len(auth.nt_challenge_response) > 24:
        nt_hash = ntowfv2(auth.user_name, self._credential.nt_hash, auth.domain_name)
        nt_challenge = NTClientChallengeV2.unpack(auth.nt_challenge_response[16:])
        time = nt_challenge.time_stamp
        client_challenge = nt_challenge.challenge_from_client
        target_info = nt_challenge.av_pairs
        expected_nt, expected_lm, key_exchange_key = compute_response_v2(nt_hash, server_challenge, client_challenge, time, target_info)
        if channel_bindings:
            if AvId.channel_bindings not in target_info:
                raise BadBindingsError(context_msg='Acceptor bindings specified but not present in initiator response')
            expected_bindings = target_info[AvId.channel_bindings]
            actual_bindings = md5(channel_bindings.pack())
            if expected_bindings not in [actual_bindings, b'\x00' * 16]:
                raise BadBindingsError(context_msg='Acceptor bindings do not match initiator bindings')
        if target_info.get(AvId.flags, 0) & AvFlags.mic:
            expected_mic = auth.mic
    else:
        if not self._nt_v1:
            raise InvalidTokenError(context_msg='Acceptor settings are set to reject NTv1 responses')
        elif not auth.nt_challenge_response and (not self._lm):
            raise InvalidTokenError(context_msg='Acceptor settings are set to reject LM responses')
        client_challenge = b'\x00' * 8
        if auth.flags & NegotiateFlags.extended_session_security:
            client_challenge = (auth.lm_challenge_response or b'\x00' * 8)[:8]
        expected_nt, expected_lm, key_exchange_key = compute_response_v1(auth.flags, self._credential.nt_hash, self._credential.lm_hash, server_challenge, client_challenge, no_lm_response=not self._lm)
    auth_success = False
    if auth.nt_challenge_response:
        auth_success = auth.nt_challenge_response == expected_nt
    elif auth.lm_challenge_response:
        auth_success = auth.lm_challenge_response == expected_lm
    if not auth_success:
        raise InvalidTokenError(context_msg='Invalid NTLM response from initiator')
    if auth.flags & NegotiateFlags.key_exch and (auth.flags & NegotiateFlags.sign or auth.flags & NegotiateFlags.seal):
        self._session_key = rc4k(key_exchange_key, auth.encrypted_random_session_key or b'')
    else:
        self._session_key = key_exchange_key
    if expected_mic:
        auth.mic = b'\x00' * 16
        actual_mic = self._calculate_mic(self.session_key, negotiate.pack(), challenge.pack(), auth.pack())
        if actual_mic != expected_mic:
            raise InvalidTokenError(context_msg='Invalid MIC in NTLM authentication message')
    self._context_attr = auth.flags
    self._complete = True