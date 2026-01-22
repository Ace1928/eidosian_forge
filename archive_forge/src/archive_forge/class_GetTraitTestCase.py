import unittest
from traits.api import (
class GetTraitTestCase(unittest.TestCase):

    def test_trait_set_bad(self):
        b = Foo(num=23)
        with self.assertRaises(TraitError):
            b.num = 'first'
        self.assertEqual(b.num, 23)

    def test_trait_set_replaced(self):
        b = Foo()
        b.add_trait('num', Str())
        b.num = 'first'
        self.assertEqual(b.num, 'first')

    def test_trait_set_replaced_and_check(self):
        b = Foo()
        b.add_trait('num', Str())
        b.num = 'first'
        self.assertEqual(b.num, 'first')
        self.assertEqual(b.trait('num'), b.traits()['num'])

    def test_trait_names_returned_by_visible_traits(self):
        b = Bar()
        self.assertEqual(sorted(b.visible_traits()), sorted(['PubT1', 'PrivT2']))

    def test_dir(self):
        b = FooBar()
        names = dir(b)
        self.assertIn('baz', names)
        self.assertIn('num', names)
        self.assertIn('edit_traits', names)
        self.assertIn('_notifiers', names)
        self.assertEqual(len(set(names)), len(names))

    def test_trait_name_with_list_items(self):

        class Base(HasTraits):
            pass
        a = Base()
        a.add_trait('pins', List())
        self.assertIn('pins', a.traits())
        self.assertNotIn('pins_items', a.traits())

    def test_trait_name_with_items(self):

        class Base(HasTraits):
            pass
        a = Base()
        a.add_trait('good_items', Str())
        self.assertNotIn('good_items', a.traits())