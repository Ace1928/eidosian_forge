import matplotlib._type1font as t1f
import os.path
import difflib
import pytest
def test_tokenize_errors():
    with pytest.raises(ValueError):
        list(t1f._tokenize(b'1234 (this (string) is unterminated\\)', True))
    with pytest.raises(ValueError):
        list(t1f._tokenize(b'/Foo<01234', True))
    with pytest.raises(ValueError):
        list(t1f._tokenize(b'/Foo<01234abcg>/Bar', True))