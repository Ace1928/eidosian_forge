from typing import Optional, Iterator, Tuple, List
from parso.python.tokenize import tokenize
from parso.utils import parse_version_string
from parso.python.token import PythonTokenTypes
def _parse_items(self):
    a, b = self._parse_item()
    while self.type in (PythonTokenTypes.NAME, PythonTokenTypes.STRING) or self.value in ('(', '['):
        c, d = self._parse_item()
        b.add_arc(c)
        b = d
    return (a, b)