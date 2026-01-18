from __future__ import annotations
import itertools
import pickle
from typing import Any
from unittest.mock import patch, Mock
from datetime import datetime, date, timedelta
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
import pytest
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points, strip_math
def verify_pre_post_state(obj):
    assert obj.meth is not obj.meth
    assert obj.aardvark is obj.aardvark
    assert a.aardvark == 'aardvark'
    assert obj.prop is obj.prop
    assert obj.cls_level is A.cls_level
    assert obj.override == 'override'
    assert not hasattr(obj, 'extra')
    assert obj.prop == 'p'
    assert obj.monkey == other.meth
    assert obj.cls_level is A.cls_level
    assert 'cls_level' not in obj.__dict__
    assert 'classy' not in obj.__dict__
    assert 'static' not in obj.__dict__