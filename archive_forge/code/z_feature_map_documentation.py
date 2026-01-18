from typing import Callable, Optional
import numpy as np
from .pauli_feature_map import PauliFeatureMap
Create a new first-order Pauli-Z expansion circuit.

        Args:
            feature_dimension: The number of features
            reps: The number of repeated circuits. Defaults to 2, has a minimum value of 1.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
            parameter_prefix: The prefix used if default parameters are generated.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.

        