import re
import uuid
import numpy as np
import param
from ... import Tiles
from ...core import util
from ...core.element import Element
from ...core.spaces import DynamicMap
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dim_range_key
from .plot import PlotlyPlot
from .util import (
class ElementPlot(PlotlyPlot, GenericElementPlot):
    aspect = param.Parameter(default='cube', doc="\n        The aspect ratio mode of the plot. By default, a plot may\n        select its own appropriate aspect ratio but sometimes it may\n        be necessary to force a square aspect ratio (e.g. to display\n        the plot as an element of a grid). The modes 'auto' and\n        'equal' correspond to the axis modes of the same name in\n        matplotlib, a numeric value may also be passed.")
    bgcolor = param.ClassSelector(class_=(str, tuple), default=None, doc='\n        If set bgcolor overrides the background color of the axis.')
    invert_axes = param.ObjectSelector(default=False, doc='\n        Inverts the axes of the plot. Note that this parameter may not\n        always be respected by all plots but should be respected by\n        adjoined plots when appropriate.')
    invert_xaxis = param.Boolean(default=False, doc='\n        Whether to invert the plot x-axis.')
    invert_yaxis = param.Boolean(default=False, doc='\n        Whether to invert the plot y-axis.')
    invert_zaxis = param.Boolean(default=False, doc='\n        Whether to invert the plot z-axis.')
    labelled = param.List(default=['x', 'y', 'z'], doc="\n        Whether to label the 'x' and 'y' axes.")
    logx = param.Boolean(default=False, doc='\n         Whether to apply log scaling to the x-axis of the Chart.')
    logy = param.Boolean(default=False, doc='\n         Whether to apply log scaling to the y-axis of the Chart.')
    logz = param.Boolean(default=False, doc='\n         Whether to apply log scaling to the y-axis of the Chart.')
    margins = param.NumericTuple(default=(50, 50, 50, 50), doc='\n         Margins in pixel values specified as a tuple of the form\n         (left, bottom, right, top).')
    responsive = param.Boolean(default=False, doc='\n         Whether the plot should stretch to fill the available space.')
    show_legend = param.Boolean(default=False, doc='\n        Whether to show legend for the plot.')
    xaxis = param.ObjectSelector(default='bottom', objects=['top', 'bottom', 'bare', 'top-bare', 'bottom-bare', None], doc="\n        Whether and where to display the xaxis, bare options allow suppressing\n        all axis labels including ticks and xlabel. Valid options are 'top',\n        'bottom', 'bare', 'top-bare' and 'bottom-bare'.")
    xticks = param.Parameter(default=None, doc='\n        Ticks along x-axis specified as an integer, explicit list of\n        tick locations, list of tuples containing the locations.')
    yaxis = param.ObjectSelector(default='left', objects=['left', 'right', 'bare', 'left-bare', 'right-bare', None], doc="\n        Whether and where to display the yaxis, bare options allow suppressing\n        all axis labels including ticks and ylabel. Valid options are 'left',\n        'right', 'bare' 'left-bare' and 'right-bare'.")
    yticks = param.Parameter(default=None, doc='\n        Ticks along y-axis specified as an integer, explicit list of\n        tick locations, list of tuples containing the locations.')
    zlabel = param.String(default=None, doc='\n        An explicit override of the z-axis label, if set takes precedence\n        over the dimension label.')
    zticks = param.Parameter(default=None, doc='\n        Ticks along z-axis specified as an integer, explicit list of\n        tick locations, list of tuples containing the locations.')
    _style_key = None
    _per_trace = False
    _supports_geo = False
    _nonvectorized_styles = []

    def __init__(self, element, plot=None, **params):
        super().__init__(element, **params)
        self.trace_uid = str(uuid.uuid4())
        self.static = len(self.hmap) == 1 and len(self.keys) == len(self.hmap)
        self.callbacks, self.source_streams = self._construct_callbacks()

    @classmethod
    def trace_kwargs(cls, **kwargs):
        return {}

    def initialize_plot(self, ranges=None, is_geo=False):
        """
        Initializes a new plot object with the last available frame.
        """
        fig = self.generate_plot(self.keys[-1], ranges, is_geo=is_geo)
        self.drawn = True
        trigger = self._trigger
        self._trigger = []
        Stream.trigger(trigger)
        return fig

    def generate_plot(self, key, ranges, element=None, is_geo=False):
        self.prev_frame = self.current_frame
        if element is None:
            element = self._get_frame(key)
        else:
            self.current_frame = element
        if is_geo and (not self._supports_geo):
            raise ValueError(f'Elements of type {type(element)} cannot be overlaid with Tiles elements using the plotly backend')
        if element is None:
            return self.handles['fig']
        plot_opts = self.lookup_options(element, 'plot').options
        self.param.update(**{k: v for k, v in plot_opts.items() if k in self.param})
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)
        self.style = self.lookup_options(element, 'style')
        style = self.style[self.cyclic_index]
        if is_geo:
            unsupported_opts = [style_opt for style_opt in style if style_opt in self.unsupported_geo_style_opts]
            if unsupported_opts:
                raise ValueError('The following {typ} style options are not supported by the Plotly backend when overlaid on Tiles:\n    {unsupported_opts}'.format(typ=type(element).__name__, unsupported_opts=unsupported_opts))
        data = self.get_data(element, ranges, style, is_geo=is_geo)
        opts = self.graph_options(element, ranges, style, is_geo=is_geo)
        components = {'traces': [], 'images': [], 'annotations': [], 'shapes': []}
        for i, d in enumerate(data):
            datum_components = self.init_graph(d, opts, index=i, is_geo=is_geo)
            traces = datum_components.get('traces', [])
            components['traces'].extend(traces)
            if i == 0 and traces:
                traces[0]['uid'] = self.trace_uid
            for k in ['images', 'shapes', 'annotations']:
                components[k].extend(datum_components.get(k, []))
            if 'mapbox' in datum_components:
                components['mapbox'] = datum_components['mapbox']
        self.handles['components'] = components
        layout = self.init_layout(key, element, ranges, is_geo=is_geo)
        for k in ['images', 'shapes', 'annotations']:
            layout.setdefault(k, [])
            layout[k].extend(components.get(k, []))
        if 'mapbox' in components:
            merge_layout(layout.setdefault('mapbox', {}), components['mapbox'])
        self.handles['layout'] = layout
        layout['autosize'] = self.responsive
        fig = dict(data=components['traces'], layout=layout, config=dict(responsive=self.responsive))
        self.handles['fig'] = fig
        self._execute_hooks(element)
        self.drawn = True
        return fig

    def graph_options(self, element, ranges, style, is_geo=False, **kwargs):
        if self.overlay_dims:
            legend = ', '.join([d.pprint_value_string(v) for d, v in self.overlay_dims.items()])
        else:
            legend = element.label
        opts = dict(name=legend, **self.trace_kwargs(is_geo=is_geo))
        if self.trace_kwargs(is_geo=is_geo).get('type', None) in legend_trace_types:
            opts.update(showlegend=self.show_legend, legendgroup=element.group + '_' + legend)
        if self._style_key is not None:
            styles = self._apply_transforms(element, ranges, style)
            key_prefix_re = re.compile('^' + self._style_key + '_')
            styles = {key_prefix_re.sub('', k): v for k, v in styles.items()}
            opts[self._style_key] = {STYLE_ALIASES.get(k, k): v for k, v in styles.items()}
            for k in ['selectedpoints', 'visible']:
                if k in opts.get(self._style_key, {}):
                    opts[k] = opts[self._style_key].pop(k)
        else:
            opts.update({STYLE_ALIASES.get(k, k): v for k, v in style.items() if k != 'cmap'})
        return opts

    def init_graph(self, datum, options, index=0, **kwargs):
        """
        Initialize the plotly components that will represent the element

        Parameters
        ----------
        datum: dict
            An element of the data list returned by the get_data method
        options: dict
            Graph options that were returned by the graph_options method
        index: int
            Index of datum in the original list returned by the get_data method

        Returns
        -------
        dict
            Dictionary of the plotly components that represent the element.
            Keys may include:
             - 'traces': List of trace dicts
             - 'annotations': List of annotations dicts
             - 'images': List of image dicts
             - 'shapes': List of shape dicts
        """
        trace = dict(options)
        for k, v in datum.items():
            if k in trace and isinstance(trace[k], dict):
                trace[k].update(v)
            else:
                trace[k] = v
        if self._style_key and self._per_trace:
            vectorized = {k: v for k, v in options[self._style_key].items() if isinstance(v, np.ndarray)}
            trace[self._style_key] = dict(trace[self._style_key])
            for s, val in vectorized.items():
                trace[self._style_key][s] = val[index]
        return {'traces': [trace]}

    def get_data(self, element, ranges, style, is_geo=False):
        return []

    def get_aspect(self, xspan, yspan):
        """
        Computes the aspect ratio of the plot
        """
        return self.width / self.height

    def _get_axis_dims(self, element):
        """Returns the dimensions corresponding to each axis.

        Should return a list of dimensions or list of lists of
        dimensions, which will be formatted to label the axis
        and to link axes.
        """
        dims = element.dimensions()[:3]
        pad = [None] * max(3 - len(dims), 0)
        return dims + pad

    def _apply_transforms(self, element, ranges, style):
        new_style = dict(style)
        for k, v in dict(style).items():
            if isinstance(v, str):
                if k == 'marker' and v in 'xsdo':
                    continue
                elif v in element:
                    v = dim(v)
                elif any((d == v for d in self.overlay_dims)):
                    v = dim(next((d for d in self.overlay_dims if d == v)))
            if not isinstance(v, dim):
                continue
            elif not v.applies(element) and v.dimension not in self.overlay_dims:
                new_style.pop(k)
                self.param.warning('Specified {} dim transform {!r} could not be applied, as not all dimensions could be resolved.'.format(k, v))
                continue
            if len(v.ops) == 0 and v.dimension in self.overlay_dims:
                val = self.overlay_dims[v.dimension]
            else:
                val = v.apply(element, ranges=ranges, flat=True)
            if not util.isscalar(val) and len(util.unique_array(val)) == 1 and ('color' not in k):
                val = val[0]
            if not util.isscalar(val):
                if k in self._nonvectorized_styles:
                    element = type(element).__name__
                    raise ValueError('Mapping a dimension to the "{style}" style option is not supported by the {element} element using the {backend} backend. To map the "{dim}" dimension to the {style} use a groupby operation to overlay your data along the dimension.'.format(style=k, dim=v.dimension, element=element, backend=self.renderer.backend))
            numeric = isinstance(val, np.ndarray) and val.dtype.kind in 'uifMm'
            if 'color' in k and isinstance(val, np.ndarray) and numeric:
                copts = self.get_color_opts(v, element, ranges, style)
                new_style.pop('cmap', None)
                new_style.update(copts)
            new_style[k] = val
        return new_style

    def init_layout(self, key, element, ranges, is_geo=False):
        el = element.traverse(lambda x: x, [Element])
        el = el[0] if el else element
        layout = dict(title=self._format_title(key, separator=' '), plot_bgcolor=self.bgcolor, uirevision=True)
        if not self.responsive:
            layout['width'] = self.width
            layout['height'] = self.height
        extent = self.get_extents(element, ranges)
        if len(extent) == 4:
            l, b, r, t = extent
        else:
            l, b, z0, r, t, z1 = extent
        dims = self._get_axis_dims(el)
        if len(dims) > 2:
            xdim, ydim, zdim = dims
        else:
            xdim, ydim = dims
            zdim = None
        xlabel, ylabel, zlabel = self._get_axis_labels(dims)
        if self.invert_axes:
            if is_geo:
                raise ValueError('The invert_axes parameter is not supported on Tiles elements with the plotly backend')
            xlabel, ylabel = (ylabel, xlabel)
            ydim, xdim = (xdim, ydim)
            l, b, r, t = (b, l, t, r)
        if 'x' not in self.labelled:
            xlabel = ''
        if 'y' not in self.labelled:
            ylabel = ''
        if 'z' not in self.labelled:
            zlabel = ''
        xaxis = {}
        if xdim and (not is_geo):
            try:
                if any(np.isnan([r, l])):
                    r, l = (0, 1)
            except TypeError:
                pass
            xrange = [r, l] if self.invert_xaxis else [l, r]
            xaxis = dict(range=xrange, title=xlabel)
            if self.logx:
                xaxis['type'] = 'log'
                xaxis['range'] = np.log10(xaxis['range'])
            self._get_ticks(xaxis, self.xticks)
            if self.projection != '3d' and self.xaxis:
                xaxis['automargin'] = False
                if isinstance(xdim, (list, tuple)):
                    dim_str = '-'.join([f'{d.name}^{d.label}^{d.unit}' for d in xdim])
                else:
                    dim_str = f'{xdim.name}^{xdim.label}^{xdim.unit}'
                xaxis['_dim'] = dim_str
                if 'bare' in self.xaxis:
                    xaxis['ticks'] = ''
                    xaxis['showticklabels'] = False
                    xaxis['title'] = ''
                if 'top' in self.xaxis:
                    xaxis['side'] = 'top'
                else:
                    xaxis['side'] = 'bottom'
        yaxis = {}
        if ydim and (not is_geo):
            try:
                if any(np.isnan([b, t])):
                    b, t = (0, 1)
            except TypeError:
                pass
            yrange = [t, b] if self.invert_yaxis else [b, t]
            yaxis = dict(range=yrange, title=ylabel)
            if self.logy:
                yaxis['type'] = 'log'
                yaxis['range'] = np.log10(yaxis['range'])
            self._get_ticks(yaxis, self.yticks)
            if self.projection != '3d' and self.yaxis:
                yaxis['automargin'] = False
                if isinstance(ydim, (list, tuple)):
                    dim_str = '-'.join([f'{d.name}^{d.label}^{d.unit}' for d in ydim])
                else:
                    dim_str = f'{ydim.name}^{ydim.label}^{ydim.unit}'
                yaxis['_dim'] = (dim_str,)
                if 'bare' in self.yaxis:
                    yaxis['ticks'] = ''
                    yaxis['showticklabels'] = False
                    yaxis['title'] = ''
                if 'right' in self.yaxis:
                    yaxis['side'] = 'right'
                else:
                    yaxis['side'] = 'left'
        if is_geo:
            mapbox = {}
            if all((np.isfinite(v) for v in (l, b, r, t))):
                x_center = (l + r) / 2.0
                y_center = (b + t) / 2.0
                lons, lats = Tiles.easting_northing_to_lon_lat([x_center], [y_center])
                mapbox['center'] = dict(lat=lats[0], lon=lons[0])
                margin_left, margin_bottom, margin_right, margin_top = self.margins
                viewport_width = self.width - margin_left - margin_right
                viewport_height = self.height - margin_top - margin_bottom
                mapbox_tile_size = 512
                max_delta = 2 * np.pi * 6378137
                x_delta = r - l
                y_delta = t - b
                with np.errstate(divide='ignore'):
                    max_x_zoom = np.log2(max_delta / x_delta) - np.log2(mapbox_tile_size / viewport_width)
                    max_y_zoom = np.log2(max_delta / y_delta) - np.log2(mapbox_tile_size / viewport_height)
                mapbox['zoom'] = min(max_x_zoom, max_y_zoom)
            layout['mapbox'] = mapbox
        if isinstance(self.projection, str) and self.projection == '3d':
            scene = dict(xaxis=xaxis, yaxis=yaxis)
            if zdim:
                zrange = [z1, z0] if self.invert_zaxis else [z0, z1]
                zaxis = dict(range=zrange, title=zlabel)
                if self.logz:
                    zaxis['type'] = 'log'
                self._get_ticks(zaxis, self.zticks)
                scene['zaxis'] = zaxis
            if self.aspect == 'cube':
                scene['aspectmode'] = 'cube'
            else:
                scene['aspectmode'] = 'manual'
                scene['aspectratio'] = self.aspect
            layout['scene'] = scene
        else:
            l, b, r, t = self.margins
            layout['margin'] = dict(l=l, r=r, b=b, t=t, pad=4)
            if not is_geo:
                layout['xaxis'] = xaxis
                layout['yaxis'] = yaxis
        return layout

    def _get_ticks(self, axis, ticker):
        axis_props = {}
        if isinstance(ticker, (tuple, list)):
            if all((isinstance(t, tuple) for t in ticker)):
                ticks, labels = zip(*ticker)
                labels = [l if isinstance(l, str) else str(l) for l in labels]
                axis_props['tickvals'] = ticks
                axis_props['ticktext'] = labels
            else:
                axis_props['tickvals'] = ticker
            axis.update(axis_props)

    def update_frame(self, key, ranges=None, element=None, is_geo=False):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        self.generate_plot(key, ranges, element, is_geo=is_geo)