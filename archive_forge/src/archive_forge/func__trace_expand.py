from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def _trace_expand(self, tree, frontier, production):
    if self._trace > 2:
        print('Expand: %s' % production)
    if self._trace > 1:
        self._trace_tree(tree, frontier, 'E')