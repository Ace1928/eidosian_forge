from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import MultiBaseInterfaceTest
from geoviews.data.geom_dict import GeomDictInterface

    Test for the MultiInterface and GeomDictInterface.
    