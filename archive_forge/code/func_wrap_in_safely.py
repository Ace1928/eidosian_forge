from __future__ import annotations
import logging # isort:skip
from ..core.templates import SCRIPT_TAG
from ..util.strings import indent
def wrap_in_safely(code: str) -> str:
    """

    """
    return _SAFELY % dict(code=indent(code, 2))