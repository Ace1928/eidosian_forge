from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
class BoundsXYCallback(BoundsCallback):
    boundsx = True
    boundsy = True