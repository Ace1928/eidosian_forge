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
def _build_iov_list(self, iov: typing.Iterable[IOV], native_convert: typing.Callable[[IOVBuffer], NativeIOV]) -> typing.List[NativeIOV]:
    """Creates a list of IOV buffers for the native provider needed."""
    provider_iov: typing.List[NativeIOV] = []
    for entry in iov:
        data: typing.Optional[typing.Union[bytes, int, bool]]
        if isinstance(entry, tuple):
            if len(entry) != 2:
                raise ValueError('IOV entry tuple must contain 2 values, the type and data, see IOVBuffer.')
            if not isinstance(entry[0], int):
                raise ValueError('IOV entry[0] must specify the BufferType as an int')
            buffer_type = entry[0]
            if entry[1] is not None and (not isinstance(entry[1], (bytes, int, bool))):
                raise ValueError('IOV entry[1] must specify the buffer bytes, length of the buffer, or whether it is auto allocated.')
            data = entry[1] if entry[1] is not None else b''
        elif isinstance(entry, int):
            buffer_type = entry
            data = None
        elif isinstance(entry, bytes):
            buffer_type = BufferType.data
            data = entry
        else:
            raise ValueError('IOV entry must be a IOVBuffer tuple, int, or bytes')
        iov_buffer = IOVBuffer(type=BufferType(buffer_type), data=data)
        provider_iov.append(native_convert(iov_buffer))
    return provider_iov