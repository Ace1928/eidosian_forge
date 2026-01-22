import unittest
from zope.interface.tests import OptimizationTestMixin
def _addValueToLeaf(self, existing_leaf_sequence, new_item):
    if not existing_leaf_sequence:
        existing_leaf_sequence = self._leafSequenceType()
    existing_leaf_sequence.append(new_item)
    return existing_leaf_sequence