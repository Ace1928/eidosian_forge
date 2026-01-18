import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_stack_area_overlay(self):
    areas = Area([1, 2, 3]) * Area([1, 2, 3])
    stacked = Area.stack(areas)
    area1 = Area(([0, 1, 2], [1, 2, 3], [0, 0, 0]), vdims=['y', 'Baseline'])
    area2 = Area(([0, 1, 2], [2, 4, 6], [1, 2, 3]), vdims=['y', 'Baseline'])
    self.assertEqual(stacked, area1 * area2)