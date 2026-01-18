import pytest
import pandas as pd
def test_obj_ref(self):
    df = pd.DataFrame()
    flags = df.flags
    del df
    with pytest.raises(ValueError, match='object has been deleted'):
        flags.allows_duplicate_labels = True