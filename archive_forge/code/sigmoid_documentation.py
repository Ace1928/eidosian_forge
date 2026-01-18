from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import zero_init
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import get_width, partial
A dense layer, followed by a sigmoid (logistic) activation function. This
    is usually used instead of the Softmax layer as an output for multi-label
    classification.
    