from collections import defaultdict
import numpy as np
import param
from bokeh.models import CategoricalColorMapper, CustomJS, FactorRange, Range1d, Whisker
from bokeh.models.tools import BoxSelectTool
from bokeh.transform import jitter
from ...core.data import Dataset
from ...core.dimension import dimension_name
from ...core.util import dimension_sanitizer, isfinite
from ...operation import interpolate_curve
from ...util.transform import dim
from ..mixins import AreaMixin, BarsMixin, SpikesMixin
from ..util import compute_sizes, get_min_distance
from .element import ColorbarPlot, ElementPlot, LegendPlot, OverlayPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import (
from .util import categorize_array
class VectorFieldPlot(ColorbarPlot):
    arrow_heads = param.Boolean(default=True, doc='\n        Whether or not to draw arrow heads.')
    magnitude = param.ClassSelector(class_=(str, dim), doc='\n        Dimension or dimension value transform that declares the magnitude\n        of each vector. Magnitude is expected to be scaled between 0-1,\n        by default the magnitudes are rescaled relative to the minimum\n        distance between vectors, this can be disabled with the\n        rescale_lengths option.')
    padding = param.ClassSelector(default=0.05, class_=(int, float, tuple))
    pivot = param.ObjectSelector(default='mid', objects=['mid', 'tip', 'tail'], doc="\n        The point around which the arrows should pivot valid options\n        include 'mid', 'tip' and 'tail'.")
    rescale_lengths = param.Boolean(default=True, doc='\n        Whether the lengths will be rescaled to take into account the\n        smallest non-zero distance between two vectors.')
    color_index = param.ClassSelector(default=None, class_=(str, int), allow_None=True, doc="\n        Deprecated in favor of dimension value transform on color option,\n        e.g. `color=dim('Magnitude')`.\n        ")
    size_index = param.ClassSelector(default=None, class_=(str, int), allow_None=True, doc="\n        Deprecated in favor of the magnitude option, e.g.\n        `magnitude=dim('Magnitude')`.\n        ")
    normalize_lengths = param.Boolean(default=True, doc="\n        Deprecated in favor of rescaling length using dimension value\n        transforms using the magnitude option, e.g.\n        `dim('Magnitude').norm()`.")
    selection_display = BokehOverlaySelectionDisplay()
    style_opts = base_properties + line_properties + ['scale', 'cmap']
    _nonvectorized_styles = base_properties + ['scale', 'cmap']
    _plot_methods = dict(single='segment')

    def _get_lengths(self, element, ranges):
        size_dim = element.get_dimension(self.size_index)
        mag_dim = self.magnitude
        if size_dim and mag_dim:
            self.param.warning("Cannot declare style mapping for 'magnitude' option and declare a size_index; ignoring the size_index.")
        elif size_dim:
            mag_dim = size_dim
        elif isinstance(mag_dim, str):
            mag_dim = element.get_dimension(mag_dim)
        (x0, x1), (y0, y1) = (element.range(i) for i in range(2))
        if mag_dim:
            if isinstance(mag_dim, dim):
                magnitudes = mag_dim.apply(element, flat=True)
            else:
                magnitudes = element.dimension_values(mag_dim)
                _, max_magnitude = ranges[dimension_name(mag_dim)]['combined']
                if self.normalize_lengths and max_magnitude != 0:
                    magnitudes = magnitudes / max_magnitude
            if self.rescale_lengths:
                base_dist = get_min_distance(element)
                magnitudes = magnitudes * base_dist
        else:
            magnitudes = np.ones(len(element))
            if self.rescale_lengths:
                base_dist = get_min_distance(element)
                magnitudes = magnitudes * base_dist
        return magnitudes

    def _glyph_properties(self, *args):
        properties = super()._glyph_properties(*args)
        properties.pop('scale', None)
        return properties

    def get_data(self, element, ranges, style):
        input_scale = style.pop('scale', 1.0)
        rads = element.dimension_values(2)
        if self.invert_axes:
            xidx, yidx = (1, 0)
            rads = np.pi / 2 - rads
        else:
            xidx, yidx = (0, 1)
        lens = self._get_lengths(element, ranges) / input_scale
        cdim = element.get_dimension(self.color_index)
        cdata, cmapping = self._get_color_data(element, ranges, style, name='line_color')
        xs = element.dimension_values(xidx)
        ys = element.dimension_values(yidx)
        xoffsets = np.cos(rads) * lens / 2.0
        yoffsets = np.sin(rads) * lens / 2.0
        if self.pivot == 'mid':
            nxoff, pxoff = (xoffsets, xoffsets)
            nyoff, pyoff = (yoffsets, yoffsets)
        elif self.pivot == 'tip':
            nxoff, pxoff = (0, xoffsets * 2)
            nyoff, pyoff = (0, yoffsets * 2)
        elif self.pivot == 'tail':
            nxoff, pxoff = (xoffsets * 2, 0)
            nyoff, pyoff = (yoffsets * 2, 0)
        x0s, x1s = (xs + nxoff, xs - pxoff)
        y0s, y1s = (ys + nyoff, ys - pyoff)
        color = None
        if self.arrow_heads:
            arrow_len = lens / 4.0
            xa1s = x0s - np.cos(rads + np.pi / 4) * arrow_len
            ya1s = y0s - np.sin(rads + np.pi / 4) * arrow_len
            xa2s = x0s - np.cos(rads - np.pi / 4) * arrow_len
            ya2s = y0s - np.sin(rads - np.pi / 4) * arrow_len
            x0s = np.tile(x0s, 3)
            x1s = np.concatenate([x1s, xa1s, xa2s])
            y0s = np.tile(y0s, 3)
            y1s = np.concatenate([y1s, ya1s, ya2s])
            if cdim and cdim.name in cdata:
                color = np.tile(cdata[cdim.name], 3)
        elif cdim:
            color = cdata.get(cdim.name)
        data = {'x0': x0s, 'x1': x1s, 'y0': y0s, 'y1': y1s}
        mapping = dict(x0='x0', x1='x1', y0='y0', y1='y1')
        if cdim and color is not None:
            data[cdim.name] = color
            mapping.update(cmapping)
        return (data, mapping, style)