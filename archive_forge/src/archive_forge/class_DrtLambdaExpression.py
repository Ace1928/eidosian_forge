import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtLambdaExpression(DrtExpression, LambdaExpression):

    def alpha_convert(self, newvar):
        """Rename all occurrences of the variable introduced by this variable
        binder in the expression to ``newvar``.
        :param newvar: ``Variable``, for the new variable
        """
        return self.__class__(newvar, self.term.replace(self.variable, DrtVariableExpression(newvar), True))

    def fol(self):
        return LambdaExpression(self.variable, self.term.fol())

    def _pretty(self):
        variables = [self.variable]
        term = self.term
        while term.__class__ == self.__class__:
            variables.append(term.variable)
            term = term.term
        var_string = ' '.join(('%s' % v for v in variables)) + DrtTokens.DOT
        term_lines = term._pretty()
        blank = ' ' * len(var_string)
        return ['    ' + blank + line for line in term_lines[:1]] + [' \\  ' + blank + line for line in term_lines[1:2]] + [' /\\ ' + var_string + line for line in term_lines[2:3]] + ['    ' + blank + line for line in term_lines[3:]]

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        return [self.variable] + self.term.get_refs(True) if recursive else [self.variable]