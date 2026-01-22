from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
@pytest.mark.skipif(shapely is None, reason='Shapely not available')
class TestSpatialSelectColumnarShapely(TestSpatialSelectColumnar):
    __test__ = True
    method = 'shapely'