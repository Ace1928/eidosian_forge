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
def asBase64(self, maxlinelength: int=76, indent_level: int=1) -> bytes:
    return _encode_base64(self.data, maxlinelength=maxlinelength, indent_level=indent_level)