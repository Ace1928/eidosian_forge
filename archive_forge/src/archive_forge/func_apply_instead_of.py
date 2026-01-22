import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def apply_instead_of(config: Config, orig_url: str, push: bool=False) -> str:
    """Apply insteadOf / pushInsteadOf to a URL."""
    longest_needle = ''
    updated_url = orig_url
    for needle, replacement in iter_instead_of(config, push):
        if not orig_url.startswith(needle):
            continue
        if len(longest_needle) < len(needle):
            longest_needle = needle
            updated_url = replacement + orig_url[len(needle):]
    return updated_url