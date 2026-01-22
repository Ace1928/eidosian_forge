import itertools
import logging
import numbers
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring, colors, offsetbox
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.cbook import silent_list
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (Patch, Rectangle, Shadow, FancyBboxPatch,
from matplotlib.collections import (
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
from matplotlib.offsetbox import (
from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer
from . import legend_handler
def _init_legend_box(self, handles, labels, markerfirst=True):
    """
        Initialize the legend_box. The legend_box is an instance of
        the OffsetBox, which is packed with legend handles and
        texts. Once packed, their location is calculated during the
        drawing time.
        """
    fontsize = self._fontsize
    text_list = []
    handle_list = []
    handles_and_labels = []
    descent = 0.35 * fontsize * (self.handleheight - 0.7)
    height = fontsize * self.handleheight - descent
    legend_handler_map = self.get_legend_handler_map()
    for orig_handle, label in zip(handles, labels):
        handler = self.get_legend_handler(legend_handler_map, orig_handle)
        if handler is None:
            _api.warn_external(f'Legend does not support handles for {type(orig_handle).__name__} instances.\nA proxy artist may be used instead.\nSee: https://matplotlib.org/stable/users/explain/axes/legend_guide.html#controlling-the-legend-entries')
            handle_list.append(None)
        else:
            textbox = TextArea(label, multilinebaseline=True, textprops=dict(verticalalignment='baseline', horizontalalignment='left', fontproperties=self.prop))
            handlebox = DrawingArea(width=self.handlelength * fontsize, height=height, xdescent=0.0, ydescent=descent)
            text_list.append(textbox._text)
            handle_list.append(handler.legend_artist(self, orig_handle, fontsize, handlebox))
            handles_and_labels.append((handlebox, textbox))
    columnbox = []
    for handles_and_labels_column in filter(len, np.array_split(handles_and_labels, self._ncols)):
        itemboxes = [HPacker(pad=0, sep=self.handletextpad * fontsize, children=[h, t] if markerfirst else [t, h], align='baseline') for h, t in handles_and_labels_column]
        alignment = 'baseline' if markerfirst else 'right'
        columnbox.append(VPacker(pad=0, sep=self.labelspacing * fontsize, align=alignment, children=itemboxes))
    mode = 'expand' if self._mode == 'expand' else 'fixed'
    sep = self.columnspacing * fontsize
    self._legend_handle_box = HPacker(pad=0, sep=sep, align='baseline', mode=mode, children=columnbox)
    self._legend_title_box = TextArea('')
    self._legend_box = VPacker(pad=self.borderpad * fontsize, sep=self.labelspacing * fontsize, align=self._alignment, children=[self._legend_title_box, self._legend_handle_box])
    self._legend_box.set_figure(self.figure)
    self._legend_box.axes = self.axes
    self.texts = text_list
    self.legend_handles = handle_list