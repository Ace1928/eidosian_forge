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
def validate_unbounded_mode(holomaps, dynmaps):
    composite = HoloMap(enumerate(holomaps), kdims=['testing_kdim'])
    holomap_kdims = set(unique_iterator([kd.name for dm in holomaps for kd in dm.kdims]))
    hmranges = {d: composite.range(d) for d in holomap_kdims}
    if any((not {d.name for d in dm.kdims} <= holomap_kdims for dm in dynmaps)):
        raise Exception('DynamicMap that are unbounded must have key dimensions that are a subset of dimensions of the HoloMap(s) defining the keys.')
    elif not all((within_range(hmrange, dm.range(d)) for dm in dynmaps for d, hmrange in hmranges.items() if d in dm.kdims)):
        raise Exception('HoloMap(s) have keys outside the ranges specified on the DynamicMap(s).')