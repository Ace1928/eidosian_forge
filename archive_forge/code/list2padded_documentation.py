from typing import Callable, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
from ..types import List2d, Padded
Create a layer to convert a list of array inputs into Padded.