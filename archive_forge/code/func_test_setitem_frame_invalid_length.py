import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_frame_invalid_length(self, data):
    df = pd.DataFrame({'A': [1] * len(data)})
    xpr = f'Length of values \\({len(data[:5])}\\) does not match length of index \\({len(df)}\\)'
    with pytest.raises(ValueError, match=xpr):
        df['B'] = data[:5]