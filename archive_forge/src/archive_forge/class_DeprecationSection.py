import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
class DeprecationSection(_SphinxSection):
    """Parser for numpydoc "deprecation warning" sections."""

    def parse(self, text: str) -> T.Iterable[DocstringDeprecated]:
        version, desc, *_ = text.split(sep='\n', maxsplit=1) + [None, None]
        if desc is not None:
            desc = _clean_str(inspect.cleandoc(desc))
        yield DocstringDeprecated(args=[self.key], description=desc, version=_clean_str(version))