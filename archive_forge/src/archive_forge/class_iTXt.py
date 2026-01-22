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
class iTXt(str):
    """
    Subclass of string to allow iTXt chunks to look like strings while
    keeping their extra information

    """

    @staticmethod
    def __new__(cls, text, lang=None, tkey=None):
        """
        :param cls: the class to use when creating the instance
        :param text: value for this key
        :param lang: language code
        :param tkey: UTF-8 version of the key name
        """
        self = str.__new__(cls, text)
        self.lang = lang
        self.tkey = tkey
        return self