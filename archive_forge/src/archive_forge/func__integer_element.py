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
def _integer_element(value: int, ctx: SimpleNamespace) -> etree.Element:
    if -1 << 63 <= value < 1 << 64:
        el = etree.Element('integer')
        el.text = '%d' % value
        return el
    raise OverflowError(value)