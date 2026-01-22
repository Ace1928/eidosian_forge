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
def _dict_element(d: Mapping[str, PlistEncodable], ctx: SimpleNamespace) -> etree.Element:
    el = etree.Element('dict')
    items = d.items()
    if ctx.sort_keys:
        items = sorted(items)
    ctx.indent_level += 1
    for key, value in items:
        if not isinstance(key, str):
            if ctx.skipkeys:
                continue
            raise TypeError('keys must be strings')
        k = etree.SubElement(el, 'key')
        k.text = tostr(key, 'utf-8')
        el.append(_make_element(value, ctx))
    ctx.indent_level -= 1
    return el