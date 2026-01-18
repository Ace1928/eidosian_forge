import sys
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from holoviews.core.util import isscalar, unique_iterator, unique_array
from holoviews.core.data import Dataset, Interface, MultiInterface, PandasAPI
from holoviews.core.data.interface import DataError
from holoviews.core.data import PandasInterface
from holoviews.core.data.spatialpandas import get_value_array
from holoviews.core.dimension import dimension_name
from holoviews.element import Path
from ..util import asarray, geom_to_array, geom_types, geom_length
from .geom_dict import geom_from_dict

        Tests if dimension is scalar in each subpath.
        