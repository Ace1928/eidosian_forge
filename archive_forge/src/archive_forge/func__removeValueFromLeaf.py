import unittest
from zope.interface.tests import OptimizationTestMixin
def _removeValueFromLeaf(self, existing_leaf_sequence, to_remove):
    without_removed = BaseAdapterRegistry._removeValueFromLeaf(self, existing_leaf_sequence, to_remove)
    existing_leaf_sequence[:] = without_removed
    assert to_remove not in existing_leaf_sequence
    return existing_leaf_sequence