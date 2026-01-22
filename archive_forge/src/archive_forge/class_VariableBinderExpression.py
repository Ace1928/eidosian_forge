import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class VariableBinderExpression(Expression):
    """This an abstract class for any Expression that binds a variable in an
    Expression.  This includes LambdaExpressions and Quantified Expressions"""

    def __init__(self, variable, term):
        """
        :param variable: ``Variable``, for the variable
        :param term: ``Expression``, for the term
        """
        assert isinstance(variable, Variable), '%s is not a Variable' % variable
        assert isinstance(term, Expression), '%s is not an Expression' % term
        self.variable = variable
        self.term = term

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """:see: Expression.replace()"""
        assert isinstance(variable, Variable), '%s is not a Variable' % variable
        assert isinstance(expression, Expression), '%s is not an Expression' % expression
        if self.variable == variable:
            if replace_bound:
                assert isinstance(expression, AbstractVariableExpression), '%s is not a AbstractVariableExpression' % expression
                return self.__class__(expression.variable, self.term.replace(variable, expression, True, alpha_convert))
            else:
                return self
        else:
            if alpha_convert and self.variable in expression.free():
                self = self.alpha_convert(unique_variable(pattern=self.variable))
            return self.__class__(self.variable, self.term.replace(variable, expression, replace_bound, alpha_convert))

    def alpha_convert(self, newvar):
        """Rename all occurrences of the variable introduced by this variable
        binder in the expression to ``newvar``.
        :param newvar: ``Variable``, for the new variable
        """
        assert isinstance(newvar, Variable), '%s is not a Variable' % newvar
        return self.__class__(newvar, self.term.replace(self.variable, VariableExpression(newvar), True))

    def free(self):
        """:see: Expression.free()"""
        return self.term.free() - {self.variable}

    def findtype(self, variable):
        """:see Expression.findtype()"""
        assert isinstance(variable, Variable), '%s is not a Variable' % variable
        if variable == self.variable:
            return ANY_TYPE
        else:
            return self.term.findtype(variable)

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.term)])

    def visit_structured(self, function, combinator):
        """:see: Expression.visit_structured()"""
        return combinator(self.variable, function(self.term))

    def __eq__(self, other):
        """Defines equality modulo alphabetic variance.  If we are comparing
        \\x.M  and \\y.N, then check equality of M and N[x/y]."""
        if isinstance(self, other.__class__) or isinstance(other, self.__class__):
            if self.variable == other.variable:
                return self.term == other.term
            else:
                varex = VariableExpression(self.variable)
                return self.term == other.term.replace(other.variable, varex)
        else:
            return False

    def __ne__(self, other):
        return not self == other
    __hash__ = Expression.__hash__