import logging
from typing import (
from typing_extensions import TypeAlias
def _html_escape(string: str) -> str:
    """HTML escape all of these " & < >"""
    html_codes = {'"': '&quot;', '<': '&lt;', '>': '&gt;'}
    string = string.replace('&', '&amp;')
    for char in html_codes:
        string = string.replace(char, html_codes[char])
    return string