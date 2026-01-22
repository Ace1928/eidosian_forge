from functools import reduce
from nltk.parse.api import ParserI
from nltk.tree import ProbabilisticTree, Tree
def _add_constituents_spanning(self, span, constituents, tokens):
    """
        Find any constituents that might cover ``span``, and add them
        to the most likely constituents table.

        :rtype: None
        :type span: tuple(int, int)
        :param span: The section of the text for which we are
            trying to find possible constituents.  The span is
            specified as a pair of integers, where the first integer
            is the index of the first token that should be included in
            the constituent; and the second integer is the index of
            the first token that should not be included in the
            constituent.  I.e., the constituent should cover
            ``text[span[0]:span[1]]``, where ``text`` is the text
            that we are parsing.

        :type constituents: dict(tuple(int,int,Nonterminal) -> ProbabilisticToken or ProbabilisticTree)
        :param constituents: The most likely constituents table.  This
            table records the most probable tree representation for
            any given span and node value.  In particular,
            ``constituents(s,e,nv)`` is the most likely
            ``ProbabilisticTree`` that covers ``text[s:e]``
            and has a node value ``nv.symbol()``, where ``text``
            is the text that we are parsing.  When
            ``_add_constituents_spanning`` is called, ``constituents``
            should contain all possible constituents that are shorter
            than ``span``.

        :type tokens: list of tokens
        :param tokens: The text we are parsing.  This is only used for
            trace output.
        """
    changed = True
    while changed:
        changed = False
        instantiations = self._find_instantiations(span, constituents)
        for production, children in instantiations:
            subtrees = [c for c in children if isinstance(c, Tree)]
            p = reduce(lambda pr, t: pr * t.prob(), subtrees, production.prob())
            node = production.lhs().symbol()
            tree = ProbabilisticTree(node, children, prob=p)
            c = constituents.get((span[0], span[1], production.lhs()))
            if self._trace > 1:
                if c is None or c != tree:
                    if c is None or c.prob() < tree.prob():
                        print('   Insert:', end=' ')
                    else:
                        print('  Discard:', end=' ')
                    self._trace_production(production, p, span, len(tokens))
            if c is None or c.prob() < tree.prob():
                constituents[span[0], span[1], production.lhs()] = tree
                changed = True