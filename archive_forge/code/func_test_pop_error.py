import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_pop_error():
    """Check rcParams pop error."""
    with pytest.raises(TypeError, match='keys cannot be deleted.*get\\(key\\)'):
        rcParams.pop('data.load')