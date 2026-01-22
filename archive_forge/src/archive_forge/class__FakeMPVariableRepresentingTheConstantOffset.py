import collections
import numbers
class _FakeMPVariableRepresentingTheConstantOffset(object):
    """A dummy class for a singleton instance used to represent the constant.

    To represent linear expressions, we store a dictionary
    MPVariable->coefficient. To represent the constant offset of the expression,
    we use this class as a substitute: its coefficient will be the offset. To
    properly be evaluated, its solution_value() needs to be 1.
    """

    def solution_value(self):
        return 1

    def __repr__(self):
        return 'OFFSET_KEY'