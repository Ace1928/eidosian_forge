from io import (
import pytest
import pandas as pd
import pandas._testing as tm
def test_showindex_disallowed_in_kwargs():
    df = pd.DataFrame([1, 2, 3])
    with pytest.raises(ValueError, match="Pass 'index' instead of 'showindex"):
        df.to_markdown(index=True, showindex=True)