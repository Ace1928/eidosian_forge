from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_starredAssignmentErrors(self):
    """
        SyntaxErrors (not encoded in the ast) surrounding Python 3 extended
        iterable unpacking
        """
    s = ', '.join(('a%d' % i for i in range(1 << 8))) + ', *rest = range(1<<8 + 1)'
    self.flakes(s, m.TooManyExpressionsInStarredAssignment)
    s = '(' + ', '.join(('a%d' % i for i in range(1 << 8))) + ', *rest) = range(1<<8 + 1)'
    self.flakes(s, m.TooManyExpressionsInStarredAssignment)
    s = '[' + ', '.join(('a%d' % i for i in range(1 << 8))) + ', *rest] = range(1<<8 + 1)'
    self.flakes(s, m.TooManyExpressionsInStarredAssignment)
    s = ', '.join(('a%d' % i for i in range(1 << 8 + 1))) + ', *rest = range(1<<8 + 2)'
    self.flakes(s, m.TooManyExpressionsInStarredAssignment)
    s = '(' + ', '.join(('a%d' % i for i in range(1 << 8 + 1))) + ', *rest) = range(1<<8 + 2)'
    self.flakes(s, m.TooManyExpressionsInStarredAssignment)
    s = '[' + ', '.join(('a%d' % i for i in range(1 << 8 + 1))) + ', *rest] = range(1<<8 + 2)'
    self.flakes(s, m.TooManyExpressionsInStarredAssignment)
    self.flakes('\n        a, *b, *c = range(10)\n        ', m.TwoStarredExpressions)
    self.flakes('\n        a, *b, c, *d = range(10)\n        ', m.TwoStarredExpressions)
    self.flakes('\n        *a, *b, *c = range(10)\n        ', m.TwoStarredExpressions)
    self.flakes('\n        (a, *b, *c) = range(10)\n        ', m.TwoStarredExpressions)
    self.flakes('\n        (a, *b, c, *d) = range(10)\n        ', m.TwoStarredExpressions)
    self.flakes('\n        (*a, *b, *c) = range(10)\n        ', m.TwoStarredExpressions)
    self.flakes('\n        [a, *b, *c] = range(10)\n        ', m.TwoStarredExpressions)
    self.flakes('\n        [a, *b, c, *d] = range(10)\n        ', m.TwoStarredExpressions)
    self.flakes('\n        [*a, *b, *c] = range(10)\n        ', m.TwoStarredExpressions)