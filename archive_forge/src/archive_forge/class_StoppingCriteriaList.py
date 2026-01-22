import time
import warnings
from abc import ABC
from copy import deepcopy
from typing import Optional
import torch
from ..utils import add_start_docstrings, logging
class StoppingCriteriaList(list):

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any((criteria(input_ids, scores, **kwargs) for criteria in self))

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None