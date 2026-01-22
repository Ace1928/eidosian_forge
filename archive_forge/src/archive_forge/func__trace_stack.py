from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import Tree
def _trace_stack(self, stack, remaining_text, marker=' '):
    """
        Print trace output displaying the given stack and text.

        :rtype: None
        :param marker: A character that is printed to the left of the
            stack.  This is used with trace level 2 to print 'S'
            before shifted stacks and 'R' before reduced stacks.
        """
    s = '  ' + marker + ' [ '
    for elt in stack:
        if isinstance(elt, Tree):
            s += repr(Nonterminal(elt.label())) + ' '
        else:
            s += repr(elt) + ' '
    s += '* ' + ' '.join(remaining_text) + ']'
    print(s)