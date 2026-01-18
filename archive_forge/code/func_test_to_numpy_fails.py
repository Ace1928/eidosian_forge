from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_to_numpy_fails():
    ds = var * int32
    with pytest.raises(TypeError):
        to_numpy(ds)
    with pytest.raises(TypeError):
        to_numpy(Option(int32))