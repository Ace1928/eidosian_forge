import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
class BlankLineAtEnd(Rule):
    code = 392
    message = 'Blank line at end of file'

    def is_issue(self, leaf):
        return self._newline_count >= 2