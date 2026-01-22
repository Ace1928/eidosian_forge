from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
def _get_view(self):
    return ({'xlim': self.get_xlim(), 'autoscalex_on': self.get_autoscalex_on(), 'ylim': self.get_ylim(), 'autoscaley_on': self.get_autoscaley_on(), 'zlim': self.get_zlim(), 'autoscalez_on': self.get_autoscalez_on()}, (self.elev, self.azim, self.roll))