from collections import defaultdict
from typing import (
import numpy as np
from .errors import Errors
from .morphology import Morphology
from .tokens import Doc, Span, Token
from .training import Example
from .util import SimpleFrozenList, get_lang_class
class PRFScore:
    """A precision / recall / F score."""

    def __init__(self, *, tp: int=0, fp: int=0, fn: int=0) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn)

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def fscore(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * (p * r / (p + r + 1e-100))

    def to_dict(self) -> Dict[str, float]:
        return {'p': self.precision, 'r': self.recall, 'f': self.fscore}