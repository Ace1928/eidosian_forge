from typing import Callable, Tuple, cast
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ragged
from ..util import ArrayInfo
Reduce ragged-formatted sequences to their first element.