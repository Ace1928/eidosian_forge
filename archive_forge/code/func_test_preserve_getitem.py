import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_preserve_getitem(self):
    df = pd.DataFrame({'A': [1, 2]}).set_flags(allows_duplicate_labels=False)
    assert df[['A']].flags.allows_duplicate_labels is False
    assert df['A'].flags.allows_duplicate_labels is False
    assert df.loc[0].flags.allows_duplicate_labels is False
    assert df.loc[[0]].flags.allows_duplicate_labels is False
    assert df.loc[0, ['A']].flags.allows_duplicate_labels is False