from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_signature(self):
    sig = inspect.signature(DataFrame.rename)
    parameters = set(sig.parameters)
    assert parameters == {'self', 'mapper', 'index', 'columns', 'axis', 'inplace', 'copy', 'level', 'errors'}