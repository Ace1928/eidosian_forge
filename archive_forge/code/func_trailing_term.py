from sympy.core import Basic, Integer
import operator
@property
def trailing_term(self):
    if self == ord0:
        raise ValueError('ordinal zero has no trailing term')
    return self.terms[-1]