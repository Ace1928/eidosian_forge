from __future__ import annotations
import itertools
import logging
import re
import struct
import warnings
import zlib
from enum import IntEnum
from . import Image, ImageChops, ImageFile, ImagePalette, ImageSequence
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from ._binary import o32be as o32
def add_itxt(self, key, value, lang='', tkey='', zip=False):
    """Appends an iTXt chunk.

        :param key: latin-1 encodable text key name
        :param value: value for this key
        :param lang: language code
        :param tkey: UTF-8 version of the key name
        :param zip: compression flag

        """
    if not isinstance(key, bytes):
        key = key.encode('latin-1', 'strict')
    if not isinstance(value, bytes):
        value = value.encode('utf-8', 'strict')
    if not isinstance(lang, bytes):
        lang = lang.encode('utf-8', 'strict')
    if not isinstance(tkey, bytes):
        tkey = tkey.encode('utf-8', 'strict')
    if zip:
        self.add(b'iTXt', key + b'\x00\x01\x00' + lang + b'\x00' + tkey + b'\x00' + zlib.compress(value))
    else:
        self.add(b'iTXt', key + b'\x00\x00\x00' + lang + b'\x00' + tkey + b'\x00' + value)