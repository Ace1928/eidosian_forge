import math
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from inspect import signature
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
from ..base import clone
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import pairwise_kernels
from ..utils.validation import _num_samples
class GenericKernelMixin:
    """Mixin for kernels which operate on generic objects such as variable-
    length sequences, trees, and graphs.

    .. versionadded:: 0.22
    """

    @property
    def requires_vector_input(self):
        """Whether the kernel works only on fixed-length feature vectors."""
        return False