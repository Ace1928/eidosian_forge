import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
@total_ordering
class AbstractVariableExpression(Expression):
    """This class represents a variable to be used as a predicate or entity"""

    def __init__(self, variable):
        """
        :param variable: ``Variable``, for the variable
        """
        assert isinstance(variable, Variable), '%s is not a Variable' % variable
        self.variable = variable

    def simplify(self):
        return self

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """:see: Expression.replace()"""
        assert isinstance(variable, Variable), '%s is not an Variable' % variable
        assert isinstance(expression, Expression), '%s is not an Expression' % expression
        if self.variable == variable:
            return expression
        else:
            return self

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)
        if signature is None:
            signature = defaultdict(list)
        resolution = other_type
        for varEx in signature[self.variable.name]:
            resolution = varEx.type.resolve(resolution)
            if not resolution:
                raise InconsistentTypeHierarchyException(self)
        signature[self.variable.name].append(self)
        for varEx in signature[self.variable.name]:
            varEx.type = resolution

    def findtype(self, variable):
        """:see Expression.findtype()"""
        assert isinstance(variable, Variable), '%s is not a Variable' % variable
        if self.variable == variable:
            return self.type
        else:
            return ANY_TYPE

    def predicates(self):
        """:see: Expression.predicates()"""
        return set()

    def __eq__(self, other):
        """Allow equality between instances of ``AbstractVariableExpression``
        subtypes."""
        return isinstance(other, AbstractVariableExpression) and self.variable == other.variable

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, AbstractVariableExpression):
            raise TypeError
        return self.variable < other.variable
    __hash__ = Expression.__hash__

    def __str__(self):
        return '%s' % self.variable