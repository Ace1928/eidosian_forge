from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
def _formula_tree(self, plugging, node):
    if node in plugging:
        return self._formula_tree(plugging, plugging[node])
    elif node in self.fragments:
        pred, args = self.fragments[node]
        children = [self._formula_tree(plugging, arg) for arg in args]
        return reduce(Constants.MAP[pred.variable.name], children)
    else:
        return node