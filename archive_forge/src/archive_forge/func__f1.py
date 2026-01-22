import pytest
from pandas.util._decorators import deprecate_kwarg
import pandas._testing as tm
@deprecate_kwarg('old', 'new')
def _f1(new=False):
    return new