from typing import Callable, List, Tuple
from thinc.api import Model, to_numpy
from thinc.types import Ints1d, Ragged
from ..util import registry
Construct a flat array that has the indices we want to extract from the
    source data. For instance, if we want the spans (5, 9), (8, 10) the
    indices will be [5, 6, 7, 8, 8, 9].
    