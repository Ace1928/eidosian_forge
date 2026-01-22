from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class QidIter(cirq.Qid):

    @property
    def dimension(self) -> int:
        return 2

    def _comparison_key(self) -> Any:
        return 1

    def __iter__(self):
        raise NotImplementedError()