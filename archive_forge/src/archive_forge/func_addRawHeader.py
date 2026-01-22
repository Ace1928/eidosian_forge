from collections.abc import Sequence as _Sequence
from typing import (
from twisted.python.compat import cmp, comparable
def addRawHeader(self, name: Union[str, bytes], value: Union[str, bytes]) -> None:
    """
        Add a new raw value for the given header.

        @param name: The name of the header for which to set the value.

        @param value: The value to set for the named header.
        """
    if not isinstance(name, (bytes, str)):
        raise TypeError(f'Header name is an instance of {type(name)!r}, not bytes or str')
    if not isinstance(value, (bytes, str)):
        raise TypeError('Header value is an instance of %r, not bytes or str' % (type(value),))
    self._rawHeaders.setdefault(_sanitizeLinearWhitespace(self._encodeName(name)), []).append(_sanitizeLinearWhitespace(value.encode('utf8') if isinstance(value, str) else value))