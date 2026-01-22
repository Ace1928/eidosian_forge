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
def _calculate_mic(self, session_key: bytes, negotiate: bytes, challenge: bytes, authenticate: bytes) -> bytes:
    """Calculates the MIC value for the negotiated context."""
    return hmac_md5(session_key, negotiate + challenge + authenticate)