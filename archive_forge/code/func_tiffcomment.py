from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def tiffcomment(arg: str | os.PathLike[Any] | FileHandle | BinaryIO, /, comment: str | bytes | None=None, pageindex: int | None=None, tagcode: int | str | None=None) -> str | None:
    """Return or replace ImageDescription value in first page of TIFF file.

    Parameters:
        arg:
            Specifies TIFF file to open.
        comment:
            7-bit ASCII string or bytes to replace existing tag value.
            The existing value is zeroed.
        pageindex:
            Index of page which ImageDescription tag value to
            read or replace. The default is 0.
        tagcode:
            Code of tag which value to read or replace.
            The default is 270 (ImageDescription).

    Returns:
        None, if `comment` is specified. Else, the current value of the
        specified tag in the specified page.


    """
    if pageindex is None:
        pageindex = 0
    if tagcode is None:
        tagcode = 270
    mode: Any = None if comment is None else 'r+'
    with TiffFile(arg, mode=mode) as tif:
        page = tif.pages[pageindex]
        if not isinstance(page, TiffPage):
            raise IndexError(f'TiffPage {pageindex} not found')
        tag = page.tags.get(tagcode, None)
        if tag is None:
            raise ValueError(f'no {TIFF.TAGS[tagcode]} tag found')
        if comment is None:
            return tag.value
        tag.overwrite(comment)
        return None