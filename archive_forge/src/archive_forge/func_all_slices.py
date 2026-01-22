import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def all_slices(self, max_index=10):
    """
        Generate all slices with bounded start, stop and step.

        Parameters
        ----------
        max_index : int
            Maximum permitted absolute value of start, stop and step.

        Yields
        ------
        s : slice
            Slice whose components are all either None, or bounded in
            absolute value by max_index.
        """
    valid_indices = [None] + list(range(-max_index, max_index + 1))
    valid_steps = [step for step in valid_indices if step != 0]
    for start in valid_indices:
        for stop in valid_indices:
            for step in valid_steps:
                yield slice(start, stop, step)