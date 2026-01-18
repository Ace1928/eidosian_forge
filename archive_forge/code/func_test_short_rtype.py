import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.epydoc import compose, parse
def test_short_rtype() -> None:
    """Test abbreviated docstring with only return type information."""
    string = 'Short description.\n\n@rtype: float'
    docstring = parse(string)
    assert compose(docstring) == string