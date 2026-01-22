import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _stream_is_misconfigured(stream: t.TextIO) -> bool:
    """A stream is misconfigured if its encoding is ASCII."""
    return is_ascii_encoding(getattr(stream, 'encoding', None) or 'ascii')