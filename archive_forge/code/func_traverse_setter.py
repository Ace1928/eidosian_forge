import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def traverse_setter(obj, attribute, value):
    """
    Traverses the object and sets the supplied attribute on the
    object. Supports Dimensioned and DimensionedPlot types.
    """
    obj.traverse(lambda x: setattr(x, attribute, value))