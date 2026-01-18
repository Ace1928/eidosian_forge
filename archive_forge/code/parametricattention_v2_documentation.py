from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ragged
from ..util import get_width
from .noop import noop
Weight inputs by similarity to a learned vector