import base64
import copy
import logging
import sys
import typing
from spnego._context import (
from spnego._credential import (
from spnego._text import to_bytes, to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import GSSError as NativeError
from spnego.exceptions import (
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
GSSAPI proxy class for GSSAPI on Linux.

    This proxy class for GSSAPI exposes GSSAPI calls into a common interface for Kerberos authentication. This context
    uses the Python gssapi library to interface with the gss_* calls to provider Kerberos.
    