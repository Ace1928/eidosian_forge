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
class PlistTarget:
    """Event handler using the ElementTree Target API that can be
    passed to a XMLParser to produce property list objects from XML.
    It is based on the CPython plistlib module's _PlistParser class,
    but does not use the expat parser.

    >>> from fontTools.misc import etree
    >>> parser = etree.XMLParser(target=PlistTarget())
    >>> result = etree.XML(
    ...     "<dict>"
    ...     "    <key>something</key>"
    ...     "    <string>blah</string>"
    ...     "</dict>",
    ...     parser=parser)
    >>> result == {"something": "blah"}
    True

    Links:
    https://github.com/python/cpython/blob/main/Lib/plistlib.py
    http://lxml.de/parsing.html#the-target-parser-interface
    """

    def __init__(self, use_builtin_types: Optional[bool]=None, dict_type: Type[MutableMapping[str, Any]]=dict) -> None:
        self.stack: List[PlistEncodable] = []
        self.current_key: Optional[str] = None
        self.root: Optional[PlistEncodable] = None
        if use_builtin_types is None:
            self._use_builtin_types = USE_BUILTIN_TYPES
        else:
            if use_builtin_types is False:
                warnings.warn('Setting use_builtin_types to False is deprecated and will be removed soon.', DeprecationWarning)
            self._use_builtin_types = use_builtin_types
        self._dict_type = dict_type

    def start(self, tag: str, attrib: Mapping[str, str]) -> None:
        self._data: List[str] = []
        handler = _TARGET_START_HANDLERS.get(tag)
        if handler is not None:
            handler(self)

    def end(self, tag: str) -> None:
        handler = _TARGET_END_HANDLERS.get(tag)
        if handler is not None:
            handler(self)

    def data(self, data: str) -> None:
        self._data.append(data)

    def close(self) -> PlistEncodable:
        if self.root is None:
            raise ValueError('No root set.')
        return self.root

    def add_object(self, value: PlistEncodable) -> None:
        if self.current_key is not None:
            stack_top = self.stack[-1]
            if not isinstance(stack_top, collections.abc.MutableMapping):
                raise ValueError('unexpected element: %r' % stack_top)
            stack_top[self.current_key] = value
            self.current_key = None
        elif not self.stack:
            self.root = value
        else:
            stack_top = self.stack[-1]
            if not isinstance(stack_top, list):
                raise ValueError('unexpected element: %r' % stack_top)
            stack_top.append(value)

    def get_data(self) -> str:
        data = ''.join(self._data)
        self._data = []
        return data