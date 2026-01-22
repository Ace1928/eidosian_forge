from __future__ import annotations
import logging  # isort:skip
import re
from typing import Any, Iterable, overload
from urllib.parse import quote_plus
def append_docstring(docstring: str | None, extra: str) -> str | None:
    """ Safely append to docstrings.

    When Python is executed with the ``-OO`` option, doc strings are removed and
    replaced the value ``None``. This function guards against appending the
    extra content in that case.

    Args:
        docstring (str or None) : The docstring to format, or None
        extra (str): the content to append if docstring is not None

    Returns:
        str or None

    """
    return None if docstring is None else docstring + extra