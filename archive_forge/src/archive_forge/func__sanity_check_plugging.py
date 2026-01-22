from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
def _sanity_check_plugging(self, plugging, node, ancestors):
    """
        Make sure that a given plugging is legal.  We recursively go through
        each node and make sure that no constraints are violated.
        We also check that all holes have been filled.
        """
    if node in self.holes:
        ancestors = [node] + ancestors
        label = plugging[node]
    else:
        label = node
    assert label in self.labels
    for c in self.constraints:
        if c.lhs == label:
            assert c.rhs in ancestors
    args = self.fragments[label][1]
    for arg in args:
        if self.is_node(arg):
            self._sanity_check_plugging(plugging, arg, [label] + ancestors)