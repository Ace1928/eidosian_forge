import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class TwoAlternativesChild(TwoAlternatives):

    def alt1(self) -> str:
        return 'alt1'

    def alt2(self) -> NoReturn:
        raise RuntimeError