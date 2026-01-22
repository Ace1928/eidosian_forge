from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
@property
def accept_html(self):
    """True if this object accepts HTML."""
    return 'text/html' in self or 'application/xhtml+xml' in self or self.accept_xhtml