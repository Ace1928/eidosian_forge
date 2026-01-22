from functools import reduce
from nltk.parse.api import ParserI
from nltk.tree import ProbabilisticTree, Tree
def _trace_production(self, production, p, span, width):
    """
        Print trace output indicating that a given production has been
        applied at a given location.

        :param production: The production that has been applied
        :type production: Production
        :param p: The probability of the tree produced by the production.
        :type p: float
        :param span: The span of the production
        :type span: tuple
        :rtype: None
        """
    str = '|' + '.' * span[0]
    str += '=' * (span[1] - span[0])
    str += '.' * (width - span[1]) + '| '
    str += '%s' % production
    if self._trace > 2:
        str = f'{str:<40} {p:12.10f} '
    print(str)