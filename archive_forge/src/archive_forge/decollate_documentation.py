from collections import namedtuple
import param
from .. import (
from ..plotting.util import initialize_dynamic
from ..streams import Derived, Stream
from . import AdjointLayout, ViewableTree
from .operation import OperationCallable

    Decollate transforms a potentially nested dynamic HoloViews object into single
    DynamicMap that returns a non-dynamic HoloViews object. All nested streams in the
    input object are copied and attached to the resulting DynamicMap.

    Args:
        hvobj: Holoviews object

    Returns:
        DynamicMap
    