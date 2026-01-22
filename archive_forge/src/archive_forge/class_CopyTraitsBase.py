import unittest
from traits.api import HasTraits, Instance, Str
class CopyTraitsBase(unittest.TestCase):
    """ Validate that copy_traits
    """
    __test__ = False

    def setUp(self):
        super().setUp()
        self.shared = Shared(s='shared')
        self.foo = Foo(shared=self.shared, s='foo')
        self.bar = Bar(shared=self.shared, foo=self.foo, s='bar')
        self.baz = Baz(shared=self.shared, bar=self.bar, s='baz')
        self.shared2 = Shared(s='shared2')
        self.foo2 = Foo(shared=self.shared2, s='foo2')
        self.bar2 = Bar(shared=self.shared2, foo=self.foo2, s='bar2')
        self.baz2 = Baz(shared=self.shared2, bar=self.bar2, s='baz2')

    def set_shared_copy(self, value):
        """ Change the copy style for the 'shared' traits. """
        self.foo.base_trait('shared').copy = value
        self.bar.base_trait('shared').copy = value
        self.baz.base_trait('shared').copy = value