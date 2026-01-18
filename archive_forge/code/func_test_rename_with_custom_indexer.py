from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_with_custom_indexer(self):

    class MyIndexer:
        pass
    ix = MyIndexer()
    ser = Series([1, 2, 3]).rename(ix)
    assert ser.name is ix