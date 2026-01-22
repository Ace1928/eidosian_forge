import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
def _parse_item(self, key: str, value: str) -> DocstringReturns:
    match = RETURN_KEY_REGEX.match(key)
    if match is not None:
        return_name = match.group('name')
        type_name = match.group('type')
    else:
        return_name = None
        type_name = None
    return DocstringReturns(args=[self.key], description=_clean_str(value), type_name=type_name, is_generator=self.is_generator, return_name=return_name)