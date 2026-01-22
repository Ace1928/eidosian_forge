import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def assertCRS(self, plot, proj='utm'):
    import cartopy
    if Version(cartopy.__version__) < Version('0.20'):
        assert plot.crs.proj4_params['proj'] == proj
    else:
        assert plot.crs.to_dict()['proj'] == proj