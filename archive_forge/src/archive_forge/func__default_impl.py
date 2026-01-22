import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def _default_impl(self, arg, kw=99):
    """Default implementation."""