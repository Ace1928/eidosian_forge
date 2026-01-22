import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class IndividualVariableExpression(AbstractVariableExpression):
    """This class represents variables that take the form of a single lowercase
    character (other than 'e') followed by zero or more digits."""

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)
        if signature is None:
            signature = defaultdict(list)
        if not other_type.matches(ENTITY_TYPE):
            raise IllegalTypeException(self, other_type, ENTITY_TYPE)
        signature[self.variable.name].append(self)

    def _get_type(self):
        return ENTITY_TYPE
    type = property(_get_type, _set_type)

    def free(self):
        """:see: Expression.free()"""
        return {self.variable}

    def constants(self):
        """:see: Expression.constants()"""
        return set()