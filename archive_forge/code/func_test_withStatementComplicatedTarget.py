from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementComplicatedTarget(self):
    """
        If the target of a C{with} statement uses any or all of the valid forms
        for that part of the grammar (See
        U{http://docs.python.org/reference/compound_stmts.html#the-with-statement}),
        the names involved are checked both for definedness and any bindings
        created are respected in the suite of the statement and afterwards.
        """
    self.flakes("\n        c = d = e = g = h = i = None\n        with open('foo') as [(a, b), c[d], e.f, g[h:i]]:\n            a, b, c, d, e, g, h, i\n        a, b, c, d, e, g, h, i\n        ")