import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def JM_char_font_flags(font, line, ch):
    flags = detect_super_script(line, ch)
    flags += mupdf.fz_font_is_italic(font) * TEXT_FONT_ITALIC
    flags += mupdf.fz_font_is_serif(font) * TEXT_FONT_SERIFED
    flags += mupdf.fz_font_is_monospaced(font) * TEXT_FONT_MONOSPACED
    flags += mupdf.fz_font_is_bold(font) * TEXT_FONT_BOLD
    return flags