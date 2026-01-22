from __future__ import annotations
import bisect
from copy import copy, deepcopy
from datetime import datetime
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MSONable
from scipy import constants
from scipy.special import comb, erfc
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
def add_m_list(self, matrix_sum, m_list):
    """
        This adds an m_list to the output_lists and updates the current
        minimum if the list is full.
        """
    if self._output_lists is None:
        self._output_lists = [[matrix_sum, m_list]]
    else:
        bisect.insort(self._output_lists, [matrix_sum, m_list])
    if self._algo == EwaldMinimizer.ALGO_BEST_FIRST and len(self._output_lists) == self._num_to_return:
        self._finished = True
    if len(self._output_lists) > self._num_to_return:
        self._output_lists.pop()
    if len(self._output_lists) == self._num_to_return:
        self._current_minimum = self._output_lists[-1][0]