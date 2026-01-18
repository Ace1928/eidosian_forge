from __future__ import annotations
import logging
from collections.abc import Mapping
import numpy as np
import rustworkx as rx
from .types import Swap, Permutation
from .util import PermutationCircuit, permutation_circuit
Perform an approximately optimal Token Swapping algorithm to implement the permutation.

        Supports partial mappings (i.e. not-permutations) for graphs with missing tokens.

        Based on the paper: Approximation and Hardness for Token Swapping by Miltzow et al. (2016)
        ArXiV: https://arxiv.org/abs/1602.05150
        and generalization based on our own work.

        Args:
          mapping: The partial mapping to implement in swaps.
          trials: The number of trials to try to perform the mapping. Minimize over the trials.
          parallel_threshold: The number of nodes in the graph beyond which the algorithm
                will use parallel processing

        Returns:
          The swaps to implement the mapping
        