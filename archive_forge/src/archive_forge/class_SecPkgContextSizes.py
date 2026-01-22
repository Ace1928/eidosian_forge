import abc
import dataclasses
import enum
import typing
import warnings
from spnego._credential import Credential
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import FeatureMissingError, NegotiateOptions, SpnegoError
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
@dataclasses.dataclass(frozen=True)
class SecPkgContextSizes:
    """Sizes of important structures used for messages.

    This dataclass exposes the sizes of important structures used in message
    support functions like wrap, wrap_iov, sign, etc. Use
    :meth:`ContextReq.query_message_sizes` to retrieve this value for an
    authenticated context.

    Currently only ``header`` is exposed but other sizes may be added in the
    future if needed.

    Attributes:
        header: The size of the header/signature of a wrapped token. This
            corresponds to cbSecurityTrailer in SecPkgContext_Sizes in SSPI and
            the size of the allocated GSS_IOV_BUFFER_TYPE_HEADER IOV buffer.
    """
    header: int