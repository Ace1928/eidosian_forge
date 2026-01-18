from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def test_signature_properties() -> None:
    sig = _UFuncSignature([['x'], ['x', 'y']], [['z']])
    assert sig.input_core_dims == (('x',), ('x', 'y'))
    assert sig.output_core_dims == (('z',),)
    assert sig.all_input_core_dims == frozenset(['x', 'y'])
    assert sig.all_output_core_dims == frozenset(['z'])
    assert sig.num_inputs == 2
    assert sig.num_outputs == 1
    assert str(sig) == '(x),(x,y)->(z)'
    assert sig.to_gufunc_string() == '(dim0),(dim0,dim1)->(dim2)'
    assert sig.to_gufunc_string(exclude_dims=set('x')) == '(dim0_0),(dim0_1,dim1)->(dim2)'
    assert _UFuncSignature([['x']]) != _UFuncSignature([['y']])