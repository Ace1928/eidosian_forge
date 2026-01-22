import base64
import logging
import struct
import typing
import spnego
from spnego._context import (
from spnego._credential import Credential, unify_credentials
from spnego._gss import GSSAPIProxy
from spnego._spnego import (
from spnego._sspi import SSPIProxy
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
def _reset_ntlm_crypto_state(self, outgoing: bool=True) -> None:
    self._context._reset_ntlm_crypto_state(outgoing=outgoing)