from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def _trace_match(self, tree, frontier, tok):
    if self._trace > 2:
        print('Match: %r' % tok)
    if self._trace > 1:
        self._trace_tree(tree, frontier, 'M')