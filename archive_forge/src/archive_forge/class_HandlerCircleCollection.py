from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
class HandlerCircleCollection(HandlerRegularPolyCollection):
    """Handler for `.CircleCollection`\\s."""

    def create_collection(self, orig_handle, sizes, offsets, offset_transform):
        return type(orig_handle)(sizes, offsets=offsets, offset_transform=offset_transform)