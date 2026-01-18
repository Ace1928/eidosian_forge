import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from ..dimension import dimension_name
from ..util import isscalar, unique_array, unique_iterator
from .interface import DataError, Interface
from .multipath import MultiInterface, ensure_ring
from .pandas import PandasInterface

        Tests if dimension is scalar in each subpath.
        