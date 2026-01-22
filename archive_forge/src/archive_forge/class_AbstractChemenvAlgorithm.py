from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
class AbstractChemenvAlgorithm(MSONable, abc.ABC):
    """
    Base class used to define a Chemenv algorithm used to identify the correct permutation for the computation
    of the Continuous Symmetry Measure.
    """

    def __init__(self, algorithm_type):
        """
        Base constructor for ChemenvAlgorithm.

        Args:
            algorithm_type (str): Type of algorithm.
        """
        self._algorithm_type = algorithm_type

    @abc.abstractmethod
    def as_dict(self):
        """A JSON-serializable dict representation of the algorithm."""

    @property
    def algorithm_type(self):
        """
        Return the type of algorithm.

        Returns:
            str: Type of the algorithm
        """
        return self._algorithm_type

    @abc.abstractmethod
    def __str__(self):
        return ''