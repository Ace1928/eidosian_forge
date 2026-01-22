import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
def _reset_newlines(self, spacing, leaf, is_comment=False):
    self._max_new_lines_in_prefix = max(self._max_new_lines_in_prefix, self._newline_count)
    wanted = self._wanted_newline_count
    if wanted is not None:
        blank_lines = self._newline_count - 1
        if wanted > blank_lines and leaf.type != 'endmarker':
            if not is_comment:
                code = 302 if wanted == 2 else 301
                message = 'expected %s blank line, found %s' % (wanted, blank_lines)
                self.add_issue(spacing, code, message)
                self._wanted_newline_count = None
        else:
            self._wanted_newline_count = None
    if not is_comment:
        wanted = self._get_wanted_blank_lines_count()
        actual = self._max_new_lines_in_prefix - 1
        val = leaf.value
        needs_lines = val == '@' and leaf.parent.type == 'decorator' or ((val == 'class' or (val == 'async' and leaf.get_next_leaf() == 'def') or (val == 'def' and self._previous_leaf != 'async')) and leaf.parent.parent.type != 'decorated')
        if needs_lines and actual < wanted:
            func_or_cls = leaf.parent
            suite = func_or_cls.parent
            if suite.type == 'decorated':
                suite = suite.parent
            if suite.children[int(suite.type == 'suite')] != func_or_cls:
                code = 302 if wanted == 2 else 301
                message = 'expected %s blank line, found %s' % (wanted, actual)
                self.add_issue(spacing, code, message)
        self._max_new_lines_in_prefix = 0
    self._newline_count = 0