from functools import reduce
from nltk.parse.api import ParserI
from nltk.tree import ProbabilisticTree, Tree
def _find_instantiations(self, span, constituents):
    """
        :return: a list of the production instantiations that cover a
            given span of the text.  A "production instantiation" is
            a tuple containing a production and a list of children,
            where the production's right hand side matches the list of
            children; and the children cover ``span``.  :rtype: list
            of ``pair`` of ``Production``, (list of
            (``ProbabilisticTree`` or token.

        :type span: tuple(int, int)
        :param span: The section of the text for which we are
            trying to find production instantiations.  The span is
            specified as a pair of integers, where the first integer
            is the index of the first token that should be covered by
            the production instantiation; and the second integer is
            the index of the first token that should not be covered by
            the production instantiation.
        :type constituents: dict(tuple(int,int,Nonterminal) -> ProbabilisticToken or ProbabilisticTree)
        :param constituents: The most likely constituents table.  This
            table records the most probable tree representation for
            any given span and node value.  See the module
            documentation for more information.
        """
    rv = []
    for production in self._grammar.productions():
        childlists = self._match_rhs(production.rhs(), span, constituents)
        for childlist in childlists:
            rv.append((production, childlist))
    return rv