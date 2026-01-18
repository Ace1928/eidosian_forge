import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
def test_returns() -> None:
    """Test parsing returns."""
    docstring = parse('\n        Short description\n        ')
    assert docstring.returns is None
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 0
    docstring = parse('\n        Short description\n        Returns\n        -------\n        type\n        ')
    assert docstring.returns is not None
    assert docstring.returns.type_name == 'type'
    assert docstring.returns.description is None
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns
    docstring = parse('\n        Short description\n        Returns\n        -------\n        int\n            description\n        ')
    assert docstring.returns is not None
    assert docstring.returns.type_name == 'int'
    assert docstring.returns.description == 'description'
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns
    docstring = parse('\n        Returns\n        -------\n        Optional[Mapping[str, List[int]]]\n            A description: with a colon\n        ')
    assert docstring.returns is not None
    assert docstring.returns.type_name == 'Optional[Mapping[str, List[int]]]'
    assert docstring.returns.description == 'A description: with a colon'
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns
    docstring = parse('\n        Short description\n        Returns\n        -------\n        int\n            description\n            with much text\n\n            even some spacing\n        ')
    assert docstring.returns is not None
    assert docstring.returns.type_name == 'int'
    assert docstring.returns.description == 'description\nwith much text\n\neven some spacing'
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns
    docstring = parse('\n        Short description\n        Returns\n        -------\n        a : int\n            description for a\n        b : str\n            description for b\n        ')
    assert docstring.returns is not None
    assert docstring.returns.type_name == 'int'
    assert docstring.returns.description == 'description for a'
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 2
    assert docstring.many_returns[0].type_name == 'int'
    assert docstring.many_returns[0].description == 'description for a'
    assert docstring.many_returns[0].return_name == 'a'
    assert docstring.many_returns[1].type_name == 'str'
    assert docstring.many_returns[1].description == 'description for b'
    assert docstring.many_returns[1].return_name == 'b'