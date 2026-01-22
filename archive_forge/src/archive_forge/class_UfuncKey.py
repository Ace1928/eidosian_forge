import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
class UfuncKey(Enum):
    CUDAFunctor = auto()
    CUDAFunctorOnOther = auto()
    CUDAFunctorOnSelf = auto()
    CPUScalar = auto()
    CPUVector = auto()
    ScalarOnly = auto()
    Generic = auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def parse(value: str) -> 'UfuncKey':
        for k, v in UfuncKey.__members__.items():
            if k == value:
                return v
        raise AssertionError(f'unknown ufunc key {value}')