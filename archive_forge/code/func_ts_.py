import os
import re
import sys
from typing import Any, Dict, List
from sphinx.errors import ExtensionError, SphinxError
from sphinx.search import SearchLanguage
from sphinx.util import import_object
def ts_(self, dict: Dict[str, int], key: str) -> int:
    if key in dict:
        return dict[key]
    return 0