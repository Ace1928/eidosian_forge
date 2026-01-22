import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
class ImplicitNode(BracketNode):
    """
    Implicit indentation after keyword arguments, default arguments,
    annotations and dict values.
    """

    def __init__(self, config, leaf, parent):
        super().__init__(config, leaf, parent)
        self.type = IndentationTypes.IMPLICIT
        next_leaf = leaf.get_next_leaf()
        if leaf == ':' and '\n' not in next_leaf.prefix and ('\r' not in next_leaf.prefix):
            self.indentation += ' '