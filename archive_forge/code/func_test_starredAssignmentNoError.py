from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_starredAssignmentNoError(self):
    """
        Python 3 extended iterable unpacking
        """
    self.flakes('\n        a, *b = range(10)\n        ')
    self.flakes('\n        *a, b = range(10)\n        ')
    self.flakes('\n        a, *b, c = range(10)\n        ')
    self.flakes('\n        (a, *b) = range(10)\n        ')
    self.flakes('\n        (*a, b) = range(10)\n        ')
    self.flakes('\n        (a, *b, c) = range(10)\n        ')
    self.flakes('\n        [a, *b] = range(10)\n        ')
    self.flakes('\n        [*a, b] = range(10)\n        ')
    self.flakes('\n        [a, *b, c] = range(10)\n        ')
    s = ', '.join(('a%d' % i for i in range(1 << 8 - 1))) + ', *rest = range(1<<8)'
    self.flakes(s)
    s = '(' + ', '.join(('a%d' % i for i in range(1 << 8 - 1))) + ', *rest) = range(1<<8)'
    self.flakes(s)
    s = '[' + ', '.join(('a%d' % i for i in range(1 << 8 - 1))) + ', *rest] = range(1<<8)'
    self.flakes(s)