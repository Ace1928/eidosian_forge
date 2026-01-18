from __future__ import annotations
from collections.abc import Callable
from contextlib import suppress
import re
from urllib.parse import quote, unquote, urlparse, urlunparse  # noqa: F401
import mdurl
from .. import _punycode
Validate URL link is allowed in output.

    This validator can prohibit more than really needed to prevent XSS.
    It's a tradeoff to keep code simple and to be secure by default.

    Note: url should be normalized at this point, and existing entities decoded.
    