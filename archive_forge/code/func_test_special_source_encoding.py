import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises
def test_special_source_encoding():
    from joblib.test.test_func_inspect_special_encoding import big5_f
    func_code, source_file, first_line = get_func_code(big5_f)
    assert first_line == 5
    assert 'def big5_f():' in func_code
    assert 'test_func_inspect_special_encoding' in source_file