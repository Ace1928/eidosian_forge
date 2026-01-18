import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_projection_as_invalid_string(self):
    with self.assertRaisesRegex(ValueError, 'Projection must be defined'):
        self.da.hvplot.image('x', 'y', projection='foo')