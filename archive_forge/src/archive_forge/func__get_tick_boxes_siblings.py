import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
def _get_tick_boxes_siblings(self, renderer):
    """
        Get the bounding boxes for this `.axis` and its siblings
        as set by `.Figure.align_xlabels` or  `.Figure.align_ylabels`.

        By default, it just gets bboxes for *self*.
        """
    name = self._get_axis_name()
    if name not in self.figure._align_label_groups:
        return ([], [])
    grouper = self.figure._align_label_groups[name]
    bboxes = []
    bboxes2 = []
    for ax in grouper.get_siblings(self.axes):
        axis = ax._axis_map[name]
        ticks_to_draw = axis._update_ticks()
        tlb, tlb2 = axis._get_ticklabel_bboxes(ticks_to_draw, renderer)
        bboxes.extend(tlb)
        bboxes2.extend(tlb2)
    return (bboxes, bboxes2)