from k-means models and quantizing vectors by comparing them with
import warnings
import numpy as np
from collections import deque
from scipy._lib._array_api import (
from scipy._lib._util import check_random_state, rng_integers
from scipy.spatial.distance import cdist
from . import _vq
def _missing_raise():
    """Raise a ClusterError when called."""
    raise ClusterError('One of the clusters is empty. Re-run kmeans with a different initialization.')