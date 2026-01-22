import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class TwoAlternativesOverride(TwoAlternatives):

    def my_method(self, arg, kw=99) -> str:
        return 'override'

    def alt1(self) -> NoReturn:
        raise RuntimeError

    def alt2(self) -> NoReturn:
        raise RuntimeError