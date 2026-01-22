from sympy.utilities.iterables import kbins
class CondVariable:
    """ A wild token that matches conditionally.

    arg   - a wild token.
    valid - an additional constraining function on a match.
    """

    def __init__(self, arg, valid):
        self.arg = arg
        self.valid = valid

    def __eq__(self, other):
        return type(self) is type(other) and self.arg == other.arg and (self.valid == other.valid)

    def __hash__(self):
        return hash((type(self), self.arg, self.valid))

    def __str__(self):
        return 'CondVariable(%s)' % str(self.arg)