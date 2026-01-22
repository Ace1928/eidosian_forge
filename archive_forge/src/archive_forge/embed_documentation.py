from typing import Callable, Dict, Optional, Tuple, TypeVar, Union, cast
from ..config import registry
from ..initializers import uniform_init
from ..model import Model
from ..types import Floats1d, Floats2d, Ints1d, Ints2d
from ..util import get_width, partial
from .array_getitem import ints_getitem
from .chain import chain
Map integers to vectors, using a fixed-size lookup table.