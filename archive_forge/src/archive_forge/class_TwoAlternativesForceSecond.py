import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class TwoAlternativesForceSecond(TwoAlternatives):

    def _do_alt1_with_my_method(self):
        return 'reverse ' + self.my_method(0, kw=0)

    @alternative(requires='my_method', implementation=_do_alt1_with_my_method)
    def alt1(self):
        """alt1 doc."""

    def alt2(self):
        return 'alt2'