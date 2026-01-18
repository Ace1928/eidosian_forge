import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas._config.config as cf
from pandas import Index
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason='repr different')
@pytest.mark.parametrize('index,expected', [(Index(['あ', 'いい', 'ううう']), "Index(['あ', 'いい', 'ううう'], dtype='object')"), (Index(['あ', 'いい', 'ううう'] * 10), "Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n       'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n       'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n       'あ', 'いい', 'ううう'],\n      dtype='object')"), (Index(['あ', 'いい', 'ううう'] * 100), "Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n       'あ',\n       ...\n       'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',\n       'ううう'],\n      dtype='object', length=300)")])
def test_string_index_repr_with_unicode_option(self, index, expected):
    with cf.option_context('display.unicode.east_asian_width', True):
        result = repr(index)
        assert result == expected