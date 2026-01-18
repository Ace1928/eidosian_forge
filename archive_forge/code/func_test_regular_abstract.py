import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def test_regular_abstract():

    class RegularAbc(metaclass=ABCMetaImplementAnyOneOf):

        @abc.abstractmethod
        def my_method(self) -> str:
            """Docstring."""
    with pytest.raises(TypeError, match='abstract'):
        RegularAbc()