import datetime
import math
import unittest
from itertools import product
import numpy as np
import pandas as pd
from holoviews import Dimension, Element
from holoviews.core.util import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PointerXY
def test_simple_alpha_beta_double_underscore_sanitized(self):
    sanitized = sanitize_identifier('α__β')
    self.assertEqual(sanitized, 'α__β')