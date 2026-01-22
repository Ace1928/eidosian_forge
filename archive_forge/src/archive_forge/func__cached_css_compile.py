from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
@lru_cache(maxsize=_MAXCACHE)
def _cached_css_compile(pattern: str, namespaces: ct.Namespaces | None, custom: ct.CustomSelectors | None, flags: int) -> cm.SoupSieve:
    """Cached CSS compile."""
    custom_selectors = process_custom(custom)
    return cm.SoupSieve(pattern, CSSParser(pattern, custom=custom_selectors, flags=flags).process_selectors(), namespaces, custom, flags)