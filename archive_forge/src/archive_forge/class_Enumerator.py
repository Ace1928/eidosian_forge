import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
class Enumerator(Serialize):

    def __init__(self) -> None:
        self.enums: Dict[Any, int] = {}

    def get(self, item) -> int:
        if item not in self.enums:
            self.enums[item] = len(self.enums)
        return self.enums[item]

    def __len__(self):
        return len(self.enums)

    def reversed(self) -> Dict[int, Any]:
        r = {v: k for k, v in self.enums.items()}
        assert len(r) == len(self.enums)
        return r