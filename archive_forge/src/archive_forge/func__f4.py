import pytest
from pandas.util._decorators import deprecate_kwarg
import pandas._testing as tm
@deprecate_kwarg('old', None)
def _f4(old=True, unchanged=True):
    return (old, unchanged)