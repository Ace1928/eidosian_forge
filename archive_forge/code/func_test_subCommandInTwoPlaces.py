from twisted.python import usage
from twisted.trial import unittest
def test_subCommandInTwoPlaces(self):
    """
        The .parent pointer is correct even when the same Options class is
        used twice.
        """

    class SubOpt(usage.Options):
        pass

    class OptFoo(usage.Options):
        subCommands = [('foo', 'f', SubOpt, 'quux')]

    class OptBar(usage.Options):
        subCommands = [('bar', 'b', SubOpt, 'quux')]
    oFoo = OptFoo()
    oFoo.parseOptions(['foo'])
    oBar = OptBar()
    oBar.parseOptions(['bar'])
    self.assertTrue(hasattr(oFoo.subOptions, 'parent'))
    self.assertTrue(hasattr(oBar.subOptions, 'parent'))
    self.failUnlessIdentical(oFoo.subOptions.parent, oFoo)
    self.failUnlessIdentical(oBar.subOptions.parent, oBar)