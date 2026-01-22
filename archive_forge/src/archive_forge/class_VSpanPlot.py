import matplotlib as mpl
import numpy as np
import pandas as pd
import param
from matplotlib import patches
from matplotlib.lines import Line2D
from ...core.options import abbreviated_exception
from ...core.util import match_spec
from ...element import HLines, HSpans, VLines, VSpans
from .element import ColorbarPlot, ElementPlot
from .plot import mpl_rc_context
class VSpanPlot(AnnotationPlot):
    """Draw a vertical span on the axis"""
    style_opts = ['alpha', 'color', 'facecolor', 'edgecolor', 'linewidth', 'linestyle', 'visible']

    def draw_annotation(self, axis, positions, opts):
        """Draw a vertical span on the axis"""
        if self.invert_axes:
            return [axis.axhspan(*positions, **opts)]
        else:
            return [axis.axvspan(*positions, **opts)]