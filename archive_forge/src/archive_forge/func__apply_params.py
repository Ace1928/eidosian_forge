import math
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine
def _apply_params(self, **kwargs):
    super()._apply_params(**kwargs)
    trans = self.label1.get_transform()
    if not trans.contains_branch(self._text1_translate):
        self.label1.set_transform(trans + self._text1_translate)
    trans = self.label2.get_transform()
    if not trans.contains_branch(self._text2_translate):
        self.label2.set_transform(trans + self._text2_translate)