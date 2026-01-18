import dill
from dill import objects
from dill import load_types
def test_typing():
    import typing
    x = typing.Any
    assert x == dill.copy(x)
    x = typing.Dict[int, str]
    assert x == dill.copy(x)
    x = typing.List[int]
    assert x == dill.copy(x)
    x = typing.Tuple[int, str]
    assert x == dill.copy(x)
    x = typing.Tuple[int]
    assert x == dill.copy(x)
    x = typing.Tuple[()]
    assert x == dill.copy(x)
    x = typing.Tuple[()].copy_with(())
    assert x == dill.copy(x)
    return