import unittest
from zope.interface.interface import Element
def test_taggedValues(self):
    """Test that we can update tagged values of more than one element
        """
    e1 = Element('foo')
    e2 = Element('bar')
    e1.setTaggedValue('x', 1)
    e2.setTaggedValue('x', 2)
    self.assertEqual(e1.getTaggedValue('x'), 1)
    self.assertEqual(e2.getTaggedValue('x'), 2)