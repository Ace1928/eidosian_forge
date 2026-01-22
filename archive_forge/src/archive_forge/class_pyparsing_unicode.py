import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
from operator import itemgetter
import itertools
from functools import wraps
from contextlib import contextmanager
class pyparsing_unicode(unicode_set):
    """
    A namespace class for defining common language unicode_sets.
    """
    _ranges = [(32, sys.maxunicode)]

    class Latin1(unicode_set):
        """Unicode set for Latin-1 Unicode Character Range"""
        _ranges = [(32, 126), (160, 255)]

    class LatinA(unicode_set):
        """Unicode set for Latin-A Unicode Character Range"""
        _ranges = [(256, 383)]

    class LatinB(unicode_set):
        """Unicode set for Latin-B Unicode Character Range"""
        _ranges = [(384, 591)]

    class Greek(unicode_set):
        """Unicode set for Greek Unicode Character Ranges"""
        _ranges = [(880, 1023), (7936, 7957), (7960, 7965), (7968, 8005), (8008, 8013), (8016, 8023), (8025,), (8027,), (8029,), (8031, 8061), (8064, 8116), (8118, 8132), (8134, 8147), (8150, 8155), (8157, 8175), (8178, 8180), (8182, 8190)]

    class Cyrillic(unicode_set):
        """Unicode set for Cyrillic Unicode Character Range"""
        _ranges = [(1024, 1279)]

    class Chinese(unicode_set):
        """Unicode set for Chinese Unicode Character Range"""
        _ranges = [(19968, 40959), (12288, 12351)]

    class Japanese(unicode_set):
        """Unicode set for Japanese Unicode Character Range, combining Kanji, Hiragana, and Katakana ranges"""
        _ranges = []

        class Kanji(unicode_set):
            """Unicode set for Kanji Unicode Character Range"""
            _ranges = [(19968, 40895), (12288, 12351)]

        class Hiragana(unicode_set):
            """Unicode set for Hiragana Unicode Character Range"""
            _ranges = [(12352, 12447)]

        class Katakana(unicode_set):
            """Unicode set for Katakana  Unicode Character Range"""
            _ranges = [(12448, 12543)]

    class Korean(unicode_set):
        """Unicode set for Korean Unicode Character Range"""
        _ranges = [(44032, 55215), (4352, 4607), (12592, 12687), (43360, 43391), (55216, 55295), (12288, 12351)]

    class CJK(Chinese, Japanese, Korean):
        """Unicode set for combined Chinese, Japanese, and Korean (CJK) Unicode Character Range"""
        pass

    class Thai(unicode_set):
        """Unicode set for Thai Unicode Character Range"""
        _ranges = [(3585, 3642), (3647, 3675)]

    class Arabic(unicode_set):
        """Unicode set for Arabic Unicode Character Range"""
        _ranges = [(1536, 1563), (1566, 1791), (1792, 1919)]

    class Hebrew(unicode_set):
        """Unicode set for Hebrew Unicode Character Range"""
        _ranges = [(1424, 1535)]

    class Devanagari(unicode_set):
        """Unicode set for Devanagari Unicode Character Range"""
        _ranges = [(2304, 2431), (43232, 43263)]