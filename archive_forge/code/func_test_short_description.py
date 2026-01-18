import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
@pytest.mark.parametrize('source, expected', [('', None), ('\n', None), ('Short description', 'Short description'), ('\nShort description\n', 'Short description'), ('\n   Short description\n', 'Short description')])
def test_short_description(source: str, expected: str) -> None:
    """Test parsing short description."""
    docstring = parse(source)
    assert docstring.short_description == expected
    assert docstring.long_description is None
    assert not docstring.meta