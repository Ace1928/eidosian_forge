import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array
def test_to_boolean_array_from_strings_invalid_string():
    with pytest.raises(ValueError, match='cannot be cast'):
        BooleanArray._from_sequence_of_strings(['donkey'], dtype='boolean')