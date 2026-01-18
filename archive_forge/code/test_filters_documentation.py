import unittest
import numpy as np
from pygsp import graphs, filters

        Test that the different methods for filter analysis, i.e. 'exact',
        'cheby', and 'lanczos', produce the same output.
        