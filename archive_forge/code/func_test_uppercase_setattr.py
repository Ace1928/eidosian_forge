import pytest
from holoviews.core.element import AttrTree
from holoviews.element.comparison import ComparisonTestCase
def test_uppercase_setattr(self):
    self.tree.C = 3
    self.assertEqual(self.tree.C, 3)