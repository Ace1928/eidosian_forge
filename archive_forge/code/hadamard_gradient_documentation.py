from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.gradients.metric_tensor import _get_aux_wire
from pennylane import transform
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from pennylane.transforms.tape_expand import expand_invalid_trainable_hadamard_gradient
from .gradient_transform import (
Post processing function for computing a hadamard gradient.