import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
class BackslashNode(IndentationNode):
    type = IndentationTypes.BACKSLASH

    def __init__(self, config, parent_indentation, containing_leaf, spacing, parent=None):
        expr_stmt = containing_leaf.search_ancestor('expr_stmt')
        if expr_stmt is not None:
            equals = expr_stmt.children[-2]
            if '\t' in config.indentation:
                self.indentation = None
            elif equals.end_pos == spacing.start_pos:
                self.indentation = parent_indentation + config.indentation
            else:
                self.indentation = ' ' * (equals.end_pos[1] + 1)
        else:
            self.indentation = parent_indentation + config.indentation
        self.bracket_indentation = self.indentation
        self.parent = parent