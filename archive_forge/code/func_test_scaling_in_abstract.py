import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import sctypes, type_info
from ..testing import suppress_warnings
from ..volumeutils import apply_read_scaling, array_from_file, array_to_file, finite_range
from .test_volumeutils import _calculate_scale
@pytest.mark.parametrize('category0, category1, overflow', [('int', 'int', False), ('uint', 'int', False), ('float', 'int', True), ('float', 'uint', True), ('complex', 'int', True), ('complex', 'uint', True)])
def test_scaling_in_abstract(category0, category1, overflow):
    for in_type in sctypes[category0]:
        for out_type in sctypes[category1]:
            if overflow:
                with suppress_warnings():
                    check_int_a2f(in_type, out_type)
            else:
                check_int_a2f(in_type, out_type)