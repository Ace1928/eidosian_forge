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
def test_urdu_five_sanitized(self):
    try:
        sanitize_identifier('۵')
    except SyntaxError as e:
        assert str(e).startswith("String '۵' cannot be sanitized")