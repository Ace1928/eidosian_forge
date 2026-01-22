from sympy.utilities.iterables import kbins
class Compound:
    """ A little class to represent an interior node in the tree

    This is analogous to SymPy.Basic for non-Atoms
    """

    def __init__(self, op, args):
        self.op = op
        self.args = args

    def __eq__(self, other):
        return type(self) is type(other) and self.op == other.op and (self.args == other.args)

    def __hash__(self):
        return hash((type(self), self.op, self.args))

    def __str__(self):
        return '%s[%s]' % (str(self.op), ', '.join(map(str, self.args)))