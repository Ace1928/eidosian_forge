import copy
from typing import Any, cast
import srsly
from ..compat import mxnet as mx
from ..optimizers import Optimizer
from ..types import ArgsKwargs, FloatsXd
from ..util import (
from .shim import Shim
Pass the inputs through to the underlying MXNet model, keeping
        track of which items in the input are tensors requiring gradients.
        If the model returns a single value, it is converted into a one-element
        tuple. Return the outputs and a callback to backpropagate.
        