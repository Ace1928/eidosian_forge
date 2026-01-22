import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class ConstantExpression(AbstractVariableExpression):
    """This class represents variables that do not take the form of a single
    character followed by zero or more digits."""
    type = ENTITY_TYPE

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)
        if signature is None:
            signature = defaultdict(list)
        if other_type == ANY_TYPE:
            resolution = ENTITY_TYPE
        else:
            resolution = other_type
            if self.type != ENTITY_TYPE:
                resolution = resolution.resolve(self.type)
        for varEx in signature[self.variable.name]:
            resolution = varEx.type.resolve(resolution)
            if not resolution:
                raise InconsistentTypeHierarchyException(self)
        signature[self.variable.name].append(self)
        for varEx in signature[self.variable.name]:
            varEx.type = resolution

    def free(self):
        """:see: Expression.free()"""
        return set()

    def constants(self):
        """:see: Expression.constants()"""
        return {self.variable}