from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
class TO:

    def __init__(self, value) -> None:
        self.value = value

    def __str__(self) -> str:
        return f'[{self.value}]'
    __repr__ = __str__

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def view(self):
        return self