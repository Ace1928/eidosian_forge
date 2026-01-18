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
def unwrap_iov(self, iov: typing.Iterable[IOV]) -> IOVUnwrapResult:
    return self._context.unwrap_iov(iov)