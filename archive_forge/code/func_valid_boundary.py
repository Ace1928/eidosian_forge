from io import StringIO, BytesIO, TextIOWrapper
from collections.abc import Mapping
import sys
import os
import urllib.parse
from email.parser import FeedParser
from email.message import Message
import html
import locale
import tempfile
import warnings
def valid_boundary(s):
    import re
    if isinstance(s, bytes):
        _vb_pattern = b'^[ -~]{0,200}[!-~]$'
    else:
        _vb_pattern = '^[ -~]{0,200}[!-~]$'
    return re.match(_vb_pattern, s)