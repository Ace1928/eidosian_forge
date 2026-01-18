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
def test_prefix_test1(self):
    prefixed = sanitize_identifier.prefixed('_some_string')
    self.assertEqual(prefixed, True)