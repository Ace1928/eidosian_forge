import pyparsing as pp
import netaddr
from functools import reduce
from operator import and_, or_
from ovs.flow.decoders import (
class OFFilter(object):
    """OFFilter represents an Open vSwitch Flow Filter.

    It is built with a filter expression string composed of logically-separated
    clauses (see ClauseExpression for details on the clause syntax).

    Args:
        expr(str): String filter expression.
    """
    w = pp.Word(pp.alphanums + '.' + ':' + '_' + '/' + '-')
    operators = pp.Literal('=') | pp.Literal('~=') | pp.Literal('<') | pp.Literal('>') | pp.Literal('!=')
    clause = w + operators + w | w
    clause.setParseAction(ClauseExpression)
    statement = pp.infixNotation(clause, [('!', 1, pp.opAssoc.RIGHT, BoolNot), ('not', 1, pp.opAssoc.RIGHT, BoolNot), ('&&', 2, pp.opAssoc.LEFT, BoolAnd), ('and', 2, pp.opAssoc.LEFT, BoolAnd), ('||', 2, pp.opAssoc.LEFT, BoolOr), ('or', 2, pp.opAssoc.LEFT, BoolOr)])

    def __init__(self, expr):
        self._filter = self.statement.parseString(expr)

    def evaluate(self, flow):
        """Evaluate whether the flow satisfies the filter.

        Args:
            flow(Flow): a openflow or datapath flow.

        Returns:
            An EvaluationResult with the result of the evaluation.
        """
        return self._filter[0].evaluate(flow)