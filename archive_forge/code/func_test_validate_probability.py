import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
@pytest.mark.parametrize('args', [('Only.+between 0 and 1', -1), ('Only.+between 0 and 1', '1.3'), ('not convert to float', 'word'), (False, '0.6'), (False, 0), (False, 1)])
def test_validate_probability(args):
    raise_error, value = args
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            _validate_probability(value)
    else:
        value = _validate_probability(value)
        assert isinstance(value, float)