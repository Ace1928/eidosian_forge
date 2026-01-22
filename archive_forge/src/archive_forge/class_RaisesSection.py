import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
class RaisesSection(_KVSection):
    """Parser for numpydoc raises sections.

    E.g. any section that looks like this:
        ValueError
            A description of what might raise ValueError
    """

    def _parse_item(self, key: str, value: str) -> DocstringRaises:
        return DocstringRaises(args=[self.key, key], description=_clean_str(value), type_name=key if len(key) > 0 else None)