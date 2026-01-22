import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
class MultiDictInterfaceTest(MultiBaseInterfaceTest):
    """
    Test of the MultiInterface.
    """
    datatype = 'multitabular'
    interface = MultiInterface
    subtype = 'dictionary'
    __test__ = True