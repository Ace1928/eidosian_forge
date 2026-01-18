from typing import Callable, List, Tuple
from thinc.api import Model, chain, with_array
from thinc.types import Floats1d, Floats2d
from ...tokens import Doc
from ...util import registry
Flattens the input to a 1-dimensional list of scores