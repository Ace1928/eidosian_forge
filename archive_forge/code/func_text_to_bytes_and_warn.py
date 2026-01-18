import os
import sys
import warnings
from typing import Any, Callable, NoReturn, Type, Union
from cryptography.hazmat.bindings.openssl.binding import Binding
def text_to_bytes_and_warn(label: str, obj: Any) -> Any:
    """
    If ``obj`` is text, emit a warning that it should be bytes instead and try
    to convert it to bytes automatically.

    :param str label: The name of the parameter from which ``obj`` was taken
        (so a developer can easily find the source of the problem and correct
        it).

    :return: If ``obj`` is the text string type, a ``bytes`` object giving the
        UTF-8 encoding of that text is returned.  Otherwise, ``obj`` itself is
        returned.
    """
    if isinstance(obj, str):
        warnings.warn(_TEXT_WARNING.format(label), category=DeprecationWarning, stacklevel=3)
        return obj.encode('utf-8')
    return obj