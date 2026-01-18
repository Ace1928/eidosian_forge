from unittest import TestCase, SkipTest
import sys
from parameterized import parameterized
import numpy as np
import pandas as pd
from holoviews.core import GridMatrix, NdOverlay
from holoviews.element import (
from hvplot import scatter_matrix
def test_offdiagonal_hexbin(self):
    sm = scatter_matrix(self.df, chart='hexbin')
    self.assertIsInstance(sm['a', 'b'], HexTiles)