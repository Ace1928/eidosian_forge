import unittest
from traits.api import (
class TestCopyableTraitNameQueries(unittest.TestCase):

    def setUp(self):
        self.foo = Foo()

    def test_type_query(self):
        names = self.foo.copyable_trait_names(**{'type': 'trait'})
        self.assertEqual(['a', 'b', 'i', 's'], sorted(names))
        names = self.foo.copyable_trait_names(**{'type': lambda t: t in ('trait', 'property')})
        self.assertEqual(['a', 'b', 'i', 'p', 's'], sorted(names))

    def test_property_query(self):
        names = self.foo.copyable_trait_names(**{'property_fields': lambda p: p and p[1].__name__ == '_set_p'})
        self.assertEqual(['p'], names)

    def test_unmodified_query(self):
        names = self.foo.copyable_trait_names(**{'is_trait_type': lambda f: f(Str)})
        self.assertEqual(['s'], names)

    def test_queries_not_combined(self):
        """ Verify that metadata is not merged with metadata to find the
            copyable traits.
        """
        eval_true = lambda x: True
        names = self.foo.copyable_trait_names(property=eval_true, type=eval_true, transient=eval_true)
        self.assertEqual(['a', 'b', 'd', 'e', 'i', 'p', 'p_ro', 'p_wo', 's', 'trait_added', 'trait_modified'], sorted(names))