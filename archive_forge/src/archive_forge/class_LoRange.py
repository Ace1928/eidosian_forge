from __future__ import annotations
import numpy as np
from .channels import DriveChannel, MeasureChannel
from .exceptions import PulseError
class LoRange:
    """Range of LO frequency."""

    def __init__(self, lower_bound: float, upper_bound: float):
        self._lb = lower_bound
        self._ub = upper_bound

    def includes(self, lo_freq: complex) -> bool:
        """Whether `lo_freq` is within the `LoRange`.

        Args:
            lo_freq: LO frequency to be validated

        Returns:
            bool: True if lo_freq is included in this range, otherwise False
        """
        if self._lb <= abs(lo_freq) <= self._ub:
            return True
        return False

    @property
    def lower_bound(self) -> float:
        """Lower bound of the LO range"""
        return self._lb

    @property
    def upper_bound(self) -> float:
        """Upper bound of the LO range"""
        return self._ub

    def __repr__(self):
        return f'{self.__class__.__name__}({self._lb:f}, {self._ub:f})'

    def __eq__(self, other):
        """Two LO ranges are the same if they are of the same type, and
        have the same frequency range

        Args:
            other (LoRange): other LoRange

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and self._ub == other._ub and (self._lb == other._lb):
            return True
        return False