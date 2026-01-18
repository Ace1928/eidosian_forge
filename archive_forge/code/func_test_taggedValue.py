import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_taggedValue(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface import taggedValue

    class ITagged(Interface):
        foo = Attribute('foo')
        bar = Attribute('bar; must eval to Boolean True if foo does')
        taggedValue('qux', 'Spam')

    class IDerived(ITagged):
        taggedValue('qux', 'Spam Spam')
        taggedValue('foo', 'bar')

    class IDerived2(IDerived):
        pass
    self.assertEqual(ITagged.getTaggedValue('qux'), 'Spam')
    self.assertRaises(KeyError, ITagged.getTaggedValue, 'foo')
    self.assertEqual(list(ITagged.getTaggedValueTags()), ['qux'])
    self.assertEqual(IDerived2.getTaggedValue('qux'), 'Spam Spam')
    self.assertEqual(IDerived2.getTaggedValue('foo'), 'bar')
    self.assertEqual(set(IDerived2.getTaggedValueTags()), {'qux', 'foo'})