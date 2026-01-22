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
@property
def _seq_num_in(self) -> int:
    if self._context_attr & NegotiateFlags.extended_session_security:
        num = self.__seq_num_in
        self.__seq_num_in += 1
    else:
        num = self.__seq_num_out
        self.__seq_num_out += 1
    return num