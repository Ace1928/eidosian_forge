import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('method', [operator.methodcaller('capitalize'), operator.methodcaller('casefold'), operator.methodcaller('cat', ['a']), operator.methodcaller('contains', 'a'), operator.methodcaller('count', 'a'), operator.methodcaller('encode', 'utf-8'), operator.methodcaller('endswith', 'a'), operator.methodcaller('extract', '(\\w)(\\d)'), operator.methodcaller('extract', '(\\w)(\\d)', expand=False), operator.methodcaller('find', 'a'), operator.methodcaller('findall', 'a'), operator.methodcaller('get', 0), operator.methodcaller('index', 'a'), operator.methodcaller('len'), operator.methodcaller('ljust', 4), operator.methodcaller('lower'), operator.methodcaller('lstrip'), operator.methodcaller('match', '\\w'), operator.methodcaller('normalize', 'NFC'), operator.methodcaller('pad', 4), operator.methodcaller('partition', 'a'), operator.methodcaller('repeat', 2), operator.methodcaller('replace', 'a', 'b'), operator.methodcaller('rfind', 'a'), operator.methodcaller('rindex', 'a'), operator.methodcaller('rjust', 4), operator.methodcaller('rpartition', 'a'), operator.methodcaller('rstrip'), operator.methodcaller('slice', 4), operator.methodcaller('slice_replace', 1, repl='a'), operator.methodcaller('startswith', 'a'), operator.methodcaller('strip'), operator.methodcaller('swapcase'), operator.methodcaller('translate', {'a': 'b'}), operator.methodcaller('upper'), operator.methodcaller('wrap', 4), operator.methodcaller('zfill', 4), operator.methodcaller('isalnum'), operator.methodcaller('isalpha'), operator.methodcaller('isdigit'), operator.methodcaller('isspace'), operator.methodcaller('islower'), operator.methodcaller('isupper'), operator.methodcaller('istitle'), operator.methodcaller('isnumeric'), operator.methodcaller('isdecimal'), operator.methodcaller('get_dummies')], ids=idfn)
def test_string_method(method):
    s = pd.Series(['a1'])
    s.attrs = {'a': 1}
    result = method(s.str)
    assert result.attrs == {'a': 1}