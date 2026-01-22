from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
@functools.lru_cache(maxsize=128)
def _special_token_regex(tokens: frozenset[str]) -> 'regex.Pattern[str]':
    inner = '|'.join((regex.escape(token) for token in tokens))
    return regex.compile(f'({inner})')