import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_setdefaults_error():
    """Check rcParams popitem error."""
    with pytest.raises(TypeError, match='Use arvizrc'):
        rcParams.setdefault('data.load', 'eager')