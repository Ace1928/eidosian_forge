import configparser
import contextlib
import locale
import logging
import pathlib
import re
import sys
from itertools import chain, groupby, repeat
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union
from pip._vendor.requests.models import Request, Response
from pip._vendor.rich.console import Console, ConsoleOptions, RenderResult
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
@staticmethod
def _iter_externally_managed_error_keys() -> Iterator[str]:
    try:
        category = locale.LC_MESSAGES
    except AttributeError:
        lang: Optional[str] = None
    else:
        lang, _ = locale.getlocale(category)
    if lang is not None:
        yield f'Error-{lang}'
        for sep in ('-', '_'):
            before, found, _ = lang.partition(sep)
            if not found:
                continue
            yield f'Error-{before}'
    yield 'Error'