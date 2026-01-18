from __future__ import annotations
from typing import Iterable, cast
from prompt_toolkit.utils import get_cwidth
from .base import (
def to_plain_text(value: AnyFormattedText) -> str:
    """
    Turn any kind of formatted text back into plain text.
    """
    return fragment_list_to_text(to_formatted_text(value))