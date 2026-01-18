from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
def test_rect_pure_split():
    pure_square = [0.0, 0.0, 1.0, 1.0]
    h_2split = [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)]
    conf_h = dict(proportion=[1, 1], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)
    v_2split = [(0.0, 0.0, 1.0, 0.5), (0.0, 0.5, 1.0, 0.5)]
    conf_v = dict(proportion=[1, 1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square, **conf_v), v_2split)
    h_2split = [(0.0, 0.0, 1 / 3, 1.0), (1 / 3, 0.0, 2 / 3, 1.0)]
    conf_h = dict(proportion=[1, 2], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)
    v_2split = [(0.0, 0.0, 1.0, 1 / 3), (0.0, 1 / 3, 1.0, 2 / 3)]
    conf_v = dict(proportion=[1, 2], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square, **conf_v), v_2split)
    h_2split = [(0.0, 0.0, 1 / 3, 1.0), (1 / 3, 0.0, 1 / 3, 1.0), (2 / 3, 0.0, 1 / 3, 1.0)]
    conf_h = dict(proportion=[1, 1, 1], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)
    v_2split = [(0.0, 0.0, 1.0, 1 / 3), (0.0, 1 / 3, 1.0, 1 / 3), (0.0, 2 / 3, 1.0, 1 / 3)]
    conf_v = dict(proportion=[1, 1, 1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square, **conf_v), v_2split)
    h_2split = [(0.0, 0.0, 1 / 4, 1.0), (1 / 4, 0.0, 1 / 2, 1.0), (3 / 4, 0.0, 1 / 4, 1.0)]
    conf_h = dict(proportion=[1, 2, 1], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)
    v_2split = [(0.0, 0.0, 1.0, 1 / 4), (0.0, 1 / 4, 1.0, 1 / 2), (0.0, 3 / 4, 1.0, 1 / 4)]
    conf_v = dict(proportion=[1, 2, 1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square, **conf_v), v_2split)
    null_square = [0.0, 0.0, 0.0, 0.0]
    conf = dict(proportion=[1, 1], gap=0.0, horizontal=True)
    eq(_split_rect(*null_square, **conf), [null_square, null_square])
    conf = dict(proportion=[1, 2], gap=1.0, horizontal=True)
    eq(_split_rect(*null_square, **conf), [null_square, null_square])