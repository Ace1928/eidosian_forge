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
def _step_accept_negotiate(self, token: bytes) -> bytes:
    """Process the Negotiate message from the initiator."""
    negotiate = Negotiate.unpack(token)
    flags = negotiate.flags | NegotiateFlags.request_target | NegotiateFlags.ntlm | NegotiateFlags.always_sign | NegotiateFlags.target_info | NegotiateFlags.target_type_server
    if flags & NegotiateFlags.unicode:
        flags &= ~NegotiateFlags.oem
    elif flags & NegotiateFlags.oem == 0:
        raise SpnegoError(ErrorCode.failure, context_msg='Neither NEGOTIATE_OEM or NEGOTIATE_UNICODE flags were set, cannot derive encoding for text fields')
    if flags & NegotiateFlags.extended_session_security:
        flags &= ~NegotiateFlags.lm_key
    server_challenge = os.urandom(8)
    target_name = to_text(socket.gethostname()).upper()
    target_info = TargetInfo()
    target_info[AvId.nb_computer_name] = target_name
    target_info[AvId.nb_domain_name] = 'WORKSTATION'
    target_info[AvId.dns_computer_name] = to_text(socket.getfqdn())
    target_info[AvId.timestamp] = FileTime.now()
    challenge = Challenge(flags, server_challenge, target_name=target_name, target_info=target_info)
    self._temp_negotiate = negotiate
    self._temp_challenge = challenge
    return challenge.pack()