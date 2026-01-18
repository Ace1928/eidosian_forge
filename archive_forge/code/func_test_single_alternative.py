import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def test_single_alternative():

    class SingleAlternative(metaclass=ABCMetaImplementAnyOneOf):

        def _default_impl(self, arg, kw=99):
            return f'default({arg}, {kw}) ' + self.alt()

        @alternative(requires='alt', implementation=_default_impl)
        def my_method(self, arg, kw=99):
            """my_method doc."""

        @abc.abstractmethod
        def alt(self) -> str:
            pass

    class SingleAlternativeChild(SingleAlternative):

        def alt(self) -> str:
            return 'alt'

    class SingleAlternativeOverride(SingleAlternative):

        def my_method(self, arg, kw=99):
            """my_method override."""
            return 'override'

        def alt(self):
            """Unneeded alternative method."""

    class SingleAlternativeGrandchild(SingleAlternativeChild):

        def alt(self):
            return 'alt2'

    class SingleAlternativeGrandchildOverride(SingleAlternativeChild):

        def my_method(self, arg, kw=99):
            """my_method override."""
            return 'override2'

        def alt(self):
            """Unneeded alternative method."""
    with pytest.raises(TypeError, match='abstract'):
        SingleAlternative()
    assert SingleAlternativeChild().my_method(1) == 'default(1, 99) alt'
    assert SingleAlternativeChild().my_method(2, kw=3) == 'default(2, 3) alt'
    assert SingleAlternativeOverride().my_method(4, kw=5) == 'override'
    assert SingleAlternativeGrandchild().my_method(6, kw=7) == 'default(6, 7) alt2'
    assert SingleAlternativeGrandchildOverride().my_method(8, kw=9) == 'override2'