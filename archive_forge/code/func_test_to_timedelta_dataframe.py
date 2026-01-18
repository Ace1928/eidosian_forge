from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
@pytest.mark.parametrize('arg', [np.arange(10).reshape(2, 5), pd.DataFrame(np.arange(10).reshape(2, 5))])
@pytest.mark.parametrize('errors', ['ignore', 'raise', 'coerce'])
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_to_timedelta_dataframe(self, arg, errors):
    with pytest.raises(TypeError, match='1-d array'):
        to_timedelta(arg, errors=errors)