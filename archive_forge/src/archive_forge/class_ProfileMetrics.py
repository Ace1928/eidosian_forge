import dataclasses
import os
from typing import Any, List
import torch
from .utils import print_once
@dataclasses.dataclass
class ProfileMetrics:
    microseconds: float = 0.0
    operators: int = 0
    fusions: int = 0
    graphs: int = 0

    def __iadd__(self, other: 'ProfileMetrics'):
        self.microseconds += other.microseconds
        self.operators += other.operators
        self.fusions += other.fusions
        return self

    def __add__(self, other: 'ProfileMetrics'):
        assert isinstance(other, ProfileMetrics)
        return ProfileMetrics(self.microseconds + other.microseconds, self.operators + other.operators, self.fusions + other.fusions)

    def __truediv__(self, other):
        if isinstance(other, int):
            other = ProfileMetrics(other, other, other)
        return ProfileMetrics(self.microseconds / max(1, other.microseconds), self.operators / max(1, other.operators), self.fusions / max(1, other.fusions))

    def __str__(self):
        return f'{self.operators:4.0%} ops {self.microseconds:4.0%} time'

    def tocsv(self):
        return [self.operators, self.microseconds]