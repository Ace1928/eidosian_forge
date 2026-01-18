import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
def test_optiontree_inheritance_flipped(self):
    """
        Tests for ordering problems manifested in issue #93
        """
    options = OptionTree(groups=['group1', 'group2'])
    opts3 = Options(kw3='value3')
    opts4 = Options(kw4='value4')
    options.MyType.Child = {'group1': opts3, 'group2': opts4}
    opts1 = Options(kw1='value1')
    opts2 = Options(kw2='value2')
    options.MyType = {'group1': opts1, 'group2': opts2}
    self.assertEqual(options.MyType.Child.options('group1').kwargs, {'kw1': 'value1', 'kw3': 'value3'})
    self.assertEqual(options.MyType.Child.options('group2').kwargs, {'kw2': 'value2', 'kw4': 'value4'})