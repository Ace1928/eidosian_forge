import fractions
import platform
import types
from typing import Any, Type
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, IS_MUSL
class TestClassGetItem:

    @pytest.mark.parametrize('cls', [np.number, np.integer, np.inexact, np.unsignedinteger, np.signedinteger, np.floating])
    def test_abc(self, cls: Type[np.number]) -> None:
        alias = cls[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    def test_abc_complexfloating(self) -> None:
        alias = np.complexfloating[Any, Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is np.complexfloating

    @pytest.mark.parametrize('arg_len', range(4))
    def test_abc_complexfloating_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len in (1, 2):
            assert np.complexfloating[arg_tup]
        else:
            match = f'Too {('few' if arg_len == 0 else 'many')} arguments'
            with pytest.raises(TypeError, match=match):
                np.complexfloating[arg_tup]

    @pytest.mark.parametrize('cls', [np.generic, np.flexible, np.character])
    def test_abc_non_numeric(self, cls: Type[np.generic]) -> None:
        with pytest.raises(TypeError):
            cls[Any]

    @pytest.mark.parametrize('code', np.typecodes['All'])
    def test_concrete(self, code: str) -> None:
        cls = np.dtype(code).type
        with pytest.raises(TypeError):
            cls[Any]

    @pytest.mark.parametrize('arg_len', range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            assert np.number[arg_tup]
        else:
            with pytest.raises(TypeError):
                np.number[arg_tup]

    def test_subscript_scalar(self) -> None:
        assert np.number[Any]