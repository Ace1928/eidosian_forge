import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
from typing import Union, List, Optional, Tuple, Set, Any, Dict
import torch
from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.utils.typing import TScalar, TVector
class FixedMetric(Metric):
    """
    Fixed metrics are verified to be the same when combined, or throw an error.

    FixedMetric is used for things like total_train_updates, which should not be
    combined across different multitasks or different workers.
    """
    __slots__ = ('_value',)

    def __init__(self, value: TScalar):
        self._value = self.as_number(value)

    def __add__(self, other: Optional['FixedMetric']) -> 'FixedMetric':
        if other is None:
            return self
        if self != other:
            raise ValueError(f'FixedMetrics not the same: {self} and {other}')
        return self

    def value(self) -> float:
        return self._value