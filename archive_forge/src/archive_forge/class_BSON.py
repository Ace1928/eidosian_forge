from __future__ import annotations
import datetime
import itertools
import os
import re
import struct
import sys
import uuid
from codecs import utf_8_decode as _utf_8_decode
from codecs import utf_8_encode as _utf_8_encode
from collections import abc as _abc
from typing import (
from bson.binary import (
from bson.code import Code
from bson.codec_options import (
from bson.datetime_ms import (
from bson.dbref import DBRef
from bson.decimal128 import Decimal128
from bson.errors import InvalidBSON, InvalidDocument, InvalidStringData
from bson.int64 import Int64
from bson.max_key import MaxKey
from bson.min_key import MinKey
from bson.objectid import ObjectId
from bson.regex import Regex
from bson.son import RE_TYPE, SON
from bson.timestamp import Timestamp
from bson.tz_util import utc
class BSON(bytes):
    """BSON (Binary JSON) data.

    .. warning:: Using this class to encode and decode BSON adds a performance
       cost. For better performance use the module level functions
       :func:`encode` and :func:`decode` instead.
    """

    @classmethod
    def encode(cls: Type[BSON], document: Mapping[str, Any], check_keys: bool=False, codec_options: CodecOptions[Any]=DEFAULT_CODEC_OPTIONS) -> BSON:
        """Encode a document to a new :class:`BSON` instance.

        A document can be any mapping type (like :class:`dict`).

        Raises :class:`TypeError` if `document` is not a mapping type,
        or contains keys that are not instances of
        :class:`str'. Raises :class:`~bson.errors.InvalidDocument`
        if `document` cannot be converted to :class:`BSON`.

        :Parameters:
          - `document`: mapping type representing a document
          - `check_keys` (optional): check if keys start with '$' or
            contain '.', raising :class:`~bson.errors.InvalidDocument` in
            either case
          - `codec_options` (optional): An instance of
            :class:`~bson.codec_options.CodecOptions`.

        .. versionchanged:: 3.0
           Replaced `uuid_subtype` option with `codec_options`.
        """
        return cls(encode(document, check_keys, codec_options))

    def decode(self, codec_options: CodecOptions[Any]=DEFAULT_CODEC_OPTIONS) -> dict[str, Any]:
        """Decode this BSON data.

        By default, returns a BSON document represented as a Python
        :class:`dict`. To use a different :class:`MutableMapping` class,
        configure a :class:`~bson.codec_options.CodecOptions`::

            >>> import collections  # From Python standard library.
            >>> import bson
            >>> from bson.codec_options import CodecOptions
            >>> data = bson.BSON.encode({'a': 1})
            >>> decoded_doc = bson.BSON(data).decode()
            <type 'dict'>
            >>> options = CodecOptions(document_class=collections.OrderedDict)
            >>> decoded_doc = bson.BSON(data).decode(codec_options=options)
            >>> type(decoded_doc)
            <class 'collections.OrderedDict'>

        :Parameters:
          - `codec_options` (optional): An instance of
            :class:`~bson.codec_options.CodecOptions`.

        .. versionchanged:: 3.0
           Removed `compile_re` option: PyMongo now always represents BSON
           regular expressions as :class:`~bson.regex.Regex` objects. Use
           :meth:`~bson.regex.Regex.try_compile` to attempt to convert from a
           BSON regular expression to a Python regular expression object.

           Replaced `as_class`, `tz_aware`, and `uuid_subtype` options with
           `codec_options`.
        """
        return decode(self, codec_options)