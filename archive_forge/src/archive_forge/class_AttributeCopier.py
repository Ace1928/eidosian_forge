from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
class AttributeCopier:

    def get_ref_artist(self):
        """
        Return the underlying artist that actually defines some properties
        (e.g., color) of this artist.
        """
        raise RuntimeError('get_ref_artist must overridden')

    def get_attribute_from_ref_artist(self, attr_name):
        getter = methodcaller('get_' + attr_name)
        prop = getter(super())
        return getter(self.get_ref_artist()) if prop == 'auto' else prop