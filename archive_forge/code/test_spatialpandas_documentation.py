from unittest import SkipTest
import numpy as np
from holoviews.core.data import (
from holoviews.core.data.interface import DataError
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
from .test_multiinterface import GeomTests

    Test of the DaskSpatialPandasInterface.
    