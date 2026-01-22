import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def _default_impl2(self, arg, kw=99):
    return f'default2({arg}, {kw}) ' + self.alt2()