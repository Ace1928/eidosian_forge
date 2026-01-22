from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
@property
def algorithm_type(self):
    """
        Return the type of algorithm.

        Returns:
            str: Type of the algorithm
        """
    return self._algorithm_type