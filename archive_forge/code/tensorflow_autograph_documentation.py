from functools import reduce
import numpy as np
import tensorflow as tf
import pennylane as qml
from pennylane.measurements import SampleMP, StateMP
from .tensorflow import (
Returns the vector-Jacobian product with given
            parameter values and output gradient dy