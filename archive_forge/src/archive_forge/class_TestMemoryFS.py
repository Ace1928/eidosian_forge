import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
class TestMemoryFS(MemoryFileSystem):
    protocol = 'testmem'
    test = [None]

    def __init__(self, **kwargs) -> None:
        self.test[0] = kwargs.pop('test', None)
        super().__init__(**kwargs)