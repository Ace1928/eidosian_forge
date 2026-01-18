from __future__ import annotations
import logging  # isort:skip
import re
from typing import Any, Iterable, overload
from urllib.parse import quote_plus
 Format a base URL with optional query arguments

    Args:
        url (str) :
            An base URL to append query arguments to
        arguments (dict or None, optional) :
            A mapping of key/value URL query arguments, or None (default: None)

    Returns:
        str

    