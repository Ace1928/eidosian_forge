from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def _trace_start(self, tree, frontier, text):
    print('Parsing %r' % ' '.join(text))
    if self._trace > 2:
        print('Start:')
    if self._trace > 1:
        self._trace_tree(tree, frontier, ' ')