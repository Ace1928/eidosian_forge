import re
from typing import Optional, List, Dict, Any, Match
from .core import Parser, InlineState
from .util import (
from .helpers import (
def _add_auto_link(self, url, text, state):
    state.append_token({'type': 'link', 'children': [{'type': 'text', 'raw': text}], 'attrs': {'url': escape_url(url)}})