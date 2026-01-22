import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class AbstractInstruction(object):
    """
    Abstract class for representing single instructions.
    """

    def out(self) -> str:
        pass

    def __str__(self) -> str:
        return self.out()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.out() == other.out()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.out())