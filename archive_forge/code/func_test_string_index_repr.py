import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas._config.config as cf
from pandas import Index
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason='repr different')
@pytest.mark.parametrize('index,expected', [(Index(['a', 'bb', 'ccc']), "Index(['a', 'bb', 'ccc'], dtype='object')"), (Index(['a', 'bb', 'ccc'] * 10), "Index(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc',\n       'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc',\n       'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],\n      dtype='object')"), (Index(['a', 'bb', 'ccc'] * 100), "Index(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',\n       ...\n       'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],\n      dtype='object', length=300)"), (Index(['あ', 'いい', 'ううう']), "Index(['あ', 'いい', 'ううう'], dtype='object')"), (Index(['あ', 'いい', 'ううう'] * 10), "Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n       'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n       'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],\n      dtype='object')"), (Index(['あ', 'いい', 'ううう'] * 100), "Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ',\n       ...\n       'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],\n      dtype='object', length=300)")])
def test_string_index_repr(self, index, expected):
    result = repr(index)
    assert result == expected