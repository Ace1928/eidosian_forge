from typing import Sequence, Callable
import itertools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import (
from .finite_difference import finite_diff
from .general_shift_rules import generate_shifted_tapes, process_shifts
from .gradient_transform import _no_trainable_grad
from .parameter_shift import _get_operation_recipe, expval_param_shift
Utility function to update the list of gradient tapes,
        the corresponding number of gradient tapes, and the processing functions