from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import Tree
def _match_rhs(self, rhs, rightmost_stack):
    """
        :rtype: bool
        :return: true if the right hand side of a CFG production
            matches the rightmost elements of the stack.  ``rhs``
            matches ``rightmost_stack`` if they are the same length,
            and each element of ``rhs`` matches the corresponding
            element of ``rightmost_stack``.  A nonterminal element of
            ``rhs`` matches any Tree whose node value is equal
            to the nonterminal's symbol.  A terminal element of ``rhs``
            matches any string whose type is equal to the terminal.
        :type rhs: list(terminal and Nonterminal)
        :param rhs: The right hand side of a CFG production.
        :type rightmost_stack: list(string and Tree)
        :param rightmost_stack: The rightmost elements of the parser's
            stack.
        """
    if len(rightmost_stack) != len(rhs):
        return False
    for i in range(len(rightmost_stack)):
        if isinstance(rightmost_stack[i], Tree):
            if not isinstance(rhs[i], Nonterminal):
                return False
            if rightmost_stack[i].label() != rhs[i].symbol():
                return False
        else:
            if isinstance(rhs[i], Nonterminal):
                return False
            if rightmost_stack[i] != rhs[i]:
                return False
    return True