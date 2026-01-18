from __future__ import annotations
import codecs
import collections
import datetime
import os
import re
import textwrap
from collections.abc import Generator, Iterable
from typing import IO, Any, TypeVar
from babel import dates, localtime
def wraptext(text: str, width: int=70, initial_indent: str='', subsequent_indent: str='') -> list[str]:
    """Simple wrapper around the ``textwrap.wrap`` function in the standard
    library. This version does not wrap lines on hyphens in words.

    :param text: the text to wrap
    :param width: the maximum line width
    :param initial_indent: string that will be prepended to the first line of
                           wrapped output
    :param subsequent_indent: string that will be prepended to all lines save
                              the first of wrapped output
    """
    wrapper = TextWrapper(width=width, initial_indent=initial_indent, subsequent_indent=subsequent_indent, break_long_words=False)
    return wrapper.wrap(text)