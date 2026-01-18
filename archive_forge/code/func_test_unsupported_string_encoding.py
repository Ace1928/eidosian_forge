from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_unsupported_string_encoding():
    with pytest.raises(ValueError):
        String(1, 'asdfasdf')