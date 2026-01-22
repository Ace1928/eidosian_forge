from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
def _find_top_hole(self):
    """
        Return the hole that will be the top of the formula tree.
        """
    top_holes = self._find_top_nodes(self.holes)
    assert len(top_holes) == 1
    return top_holes.pop()