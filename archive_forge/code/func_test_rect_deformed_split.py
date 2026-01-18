from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
def test_rect_deformed_split():
    non_pure_square = [1.0, -1.0, 1.0, 0.5]
    h_2split = [(1.0, -1.0, 0.5, 0.5), (1.5, -1.0, 0.5, 0.5)]
    conf_h = dict(proportion=[1, 1], gap=0.0, horizontal=True)
    eq(_split_rect(*non_pure_square, **conf_h), h_2split)
    v_2split = [(1.0, -1.0, 1.0, 0.25), (1.0, -0.75, 1.0, 0.25)]
    conf_v = dict(proportion=[1, 1], gap=0.0, horizontal=False)
    eq(_split_rect(*non_pure_square, **conf_v), v_2split)
    h_2split = [(1.0, -1.0, 1 / 3, 0.5), (1 + 1 / 3, -1.0, 2 / 3, 0.5)]
    conf_h = dict(proportion=[1, 2], gap=0.0, horizontal=True)
    eq(_split_rect(*non_pure_square, **conf_h), h_2split)
    v_2split = [(1.0, -1.0, 1.0, 1 / 6), (1.0, 1 / 6 - 1, 1.0, 2 / 6)]
    conf_v = dict(proportion=[1, 2], gap=0.0, horizontal=False)
    eq(_split_rect(*non_pure_square, **conf_v), v_2split)