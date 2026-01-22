import inspect
import re
import typing as T
from collections import OrderedDict, namedtuple
from enum import IntEnum
from .common import (
@staticmethod
def _build_multi_meta(section: Section, before: str, desc: str) -> DocstringMeta:
    if section.key in PARAM_KEYWORDS:
        match = GOOGLE_TYPED_ARG_REGEX.match(before)
        if match:
            arg_name, type_name = match.group(1, 2)
            if type_name.endswith(', optional'):
                is_optional = True
                type_name = type_name[:-10]
            elif type_name.endswith('?'):
                is_optional = True
                type_name = type_name[:-1]
            else:
                is_optional = False
        else:
            arg_name, type_name = (before, None)
            is_optional = None
        match = GOOGLE_ARG_DESC_REGEX.match(desc)
        default = match.group(1) if match else None
        return DocstringParam(args=[section.key, before], description=desc, arg_name=arg_name, type_name=type_name, is_optional=is_optional, default=default)
    if section.key in RETURNS_KEYWORDS | YIELDS_KEYWORDS:
        return DocstringReturns(args=[section.key, before], description=desc, type_name=before, is_generator=section.key in YIELDS_KEYWORDS)
    if section.key in RAISES_KEYWORDS:
        return DocstringRaises(args=[section.key, before], description=desc, type_name=before)
    return DocstringMeta(args=[section.key, before], description=desc)