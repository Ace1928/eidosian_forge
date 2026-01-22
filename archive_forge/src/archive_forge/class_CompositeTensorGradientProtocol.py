import abc
import sys
from tensorflow.python.framework import composite_tensor
from tensorflow.python.util import nest
@runtime_checkable
class CompositeTensorGradientProtocol(Protocol):
    """Protocol for adding gradient support to CompositeTensors."""
    __composite_gradient__: CompositeTensorGradient