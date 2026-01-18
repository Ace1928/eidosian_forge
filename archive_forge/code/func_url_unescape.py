import html
import json
import re
import urllib.parse
from tornado.util import unicode_type
import typing
from typing import Union, Any, Optional, Dict, List, Callable
def url_unescape(value: Union[str, bytes], encoding: Optional[str]='utf-8', plus: bool=True) -> Union[str, bytes]:
    """Decodes the given value from a URL.

    The argument may be either a byte or unicode string.

    If encoding is None, the result will be a byte string and this function is equivalent to
    `urllib.parse.unquote_to_bytes` if ``plus=False``.  Otherwise, the result is a unicode string in
    the specified encoding and this function is equivalent to either `urllib.parse.unquote_plus` or
    `urllib.parse.unquote` except that this function also accepts `bytes` as input.

    If ``plus`` is true (the default), plus signs will be interpreted as spaces (literal plus signs
    must be represented as "%2B").  This is appropriate for query strings and form-encoded values
    but not for the path component of a URL.  Note that this default is the reverse of Python's
    urllib module.

    .. versionadded:: 3.1
       The ``plus`` argument
    """
    if encoding is None:
        if plus:
            value = to_basestring(value).replace('+', ' ')
        return urllib.parse.unquote_to_bytes(value)
    else:
        unquote = urllib.parse.unquote_plus if plus else urllib.parse.unquote
        return unquote(to_basestring(value), encoding=encoding)