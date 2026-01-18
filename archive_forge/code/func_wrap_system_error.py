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
def wrap_system_error(error_type: typing.Type, context: typing.Optional[str]=None) -> typing.Callable[[F], F]:
    """Wraps a function that makes a native GSSAPI/SSPI syscall and convert native exceptions to a SpnegoError.

    Wraps a function that can potentially raise a WindowsError or GSSError and converts it to the common SpnegoError
    that is exposed by this library. This is to ensure the context proxy functions raise a common set of errors rather
    than a specific error for the provider. The underlying error is preserved in the SpnegoError if the user wishes to
    inspect that.

    Args:
        error_type: The native error type that need to be wrapped.
        context: An optional context message to add to the error if raised.
    """

    def decorator(func: F) -> F:

        def wrapper(*args: typing.Any, **kwargs: typing.Any) -> F:
            try:
                return func(*args, **kwargs)
            except error_type as native_err:
                raise SpnegoError(base_error=native_err, context_msg=context) from native_err
        return typing.cast(F, wrapper)
    return decorator