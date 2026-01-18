from typing import Sequence, Callable
import functools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.circuit_graph import LayerData
from pennylane.queuing import WrappedObj
from pennylane.transforms import transform
Postprocessing function for the full metric tensor.