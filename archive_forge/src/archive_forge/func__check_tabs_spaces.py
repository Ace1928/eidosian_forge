import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
def _check_tabs_spaces(self, spacing):
    if self._wrong_indentation_char in spacing.value:
        self.add_issue(spacing, 101, 'Indentation contains ' + self._indentation_type)
        return True
    return False