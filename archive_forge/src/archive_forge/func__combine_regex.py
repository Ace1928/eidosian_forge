import re
from abc import ABC, abstractmethod
from typing import List, Union
from .text import Span, Text
def _combine_regex(*regexes: str) -> str:
    """Combine a number of regexes in to a single regex.

    Returns:
        str: New regex with all regexes ORed together.
    """
    return '|'.join(regexes)