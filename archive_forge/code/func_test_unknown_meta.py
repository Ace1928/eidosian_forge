import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.google import (
def test_unknown_meta() -> None:
    """Test parsing unknown meta."""
    docstring = parse('Short desc\n\n        Unknown 0:\n            title0: content0\n\n        Args:\n            arg0: desc0\n            arg1: desc1\n\n        Unknown1:\n            title1: content1\n\n        Unknown2:\n            title2: content2\n        ')
    assert docstring.params[0].arg_name == 'arg0'
    assert docstring.params[0].description == 'desc0'
    assert docstring.params[1].arg_name == 'arg1'
    assert docstring.params[1].description == 'desc1'