import collections.abc
import re
from typing import (
import warnings
from io import BytesIO
from datetime import datetime
from base64 import b64encode, b64decode
from numbers import Integral
from types import SimpleNamespace
from functools import singledispatch
from fontTools.misc import etree
from fontTools.misc.textTools import tostr
def totree(value: PlistEncodable, sort_keys: bool=True, skipkeys: bool=False, use_builtin_types: Optional[bool]=None, pretty_print: bool=True, indent_level: int=1) -> etree.Element:
    """Convert a value derived from a plist into an XML tree.

    Args:
        value: Any kind of value to be serialized to XML.
        sort_keys: Whether keys of dictionaries should be sorted.
        skipkeys (bool): Whether to silently skip non-string dictionary
            keys.
        use_builtin_types (bool): If true, byte strings will be
            encoded in Base-64 and wrapped in a ``data`` tag; if
            false, they will be either stored as ASCII strings or an
            exception raised if they cannot be decoded as such. Defaults
            to ``True`` if not present. Deprecated.
        pretty_print (bool): Whether to indent the output.
        indent_level (int): Level of indentation when serializing.

    Returns: an ``etree`` ``Element`` object.

    Raises:
        ``TypeError``
            if non-string dictionary keys are serialized
            and ``skipkeys`` is false.
        ``ValueError``
            if non-ASCII binary data is present
            and `use_builtin_types` is false.
    """
    if use_builtin_types is None:
        use_builtin_types = USE_BUILTIN_TYPES
    else:
        use_builtin_types = use_builtin_types
    context = SimpleNamespace(sort_keys=sort_keys, skipkeys=skipkeys, use_builtin_types=use_builtin_types, pretty_print=pretty_print, indent_level=indent_level)
    return _make_element(value, context)