from typing import Callable, List, Optional, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
from ..types import Array2d, Array3d
Reshape data on the way into and out from a layer.