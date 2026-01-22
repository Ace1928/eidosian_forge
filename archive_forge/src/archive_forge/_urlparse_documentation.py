from __future__ import annotations
import ipaddress
import re
import typing
import idna
from ._exceptions import InvalidURL

    We can use a much simpler version of the stdlib urlencode here because
    we don't need to handle a bunch of different typing cases, such as bytes vs str.

    https://github.com/python/cpython/blob/b2f7b2ef0b5421e01efb8c7bee2ef95d3bab77eb/Lib/urllib/parse.py#L926

    Note that we use '%20' encoding for spaces. and '%2F  for '/'.
    This is slightly different than `requests`, but is the behaviour that browsers use.

    See
    - https://github.com/encode/httpx/issues/2536
    - https://github.com/encode/httpx/issues/2721
    - https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlencode
    