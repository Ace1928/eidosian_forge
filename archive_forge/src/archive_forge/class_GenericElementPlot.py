import uuid
import warnings
from ast import literal_eval
from collections import Counter, defaultdict
from functools import partial
from itertools import groupby, product
import numpy as np
import param
from panel.config import config
from panel.io.document import unlocked
from panel.io.notebook import push
from panel.io.state import state
from pyviz_comms import JupyterComm
from ..core import traversal, util
from ..core.data import Dataset, disable_pipeline
from ..core.element import Element, Element3D
from ..core.layout import Empty, Layout, NdLayout
from ..core.options import Compositor, SkipRendering, Store, lookup_options
from ..core.overlay import CompositeOverlay, NdOverlay, Overlay
from ..core.spaces import DynamicMap, HoloMap
from ..core.util import isfinite, stream_parameters
from ..element import Graph, Table
from ..selection import NoOpSelectionDisplay
from ..streams import RangeX, RangeXY, RangeY, Stream
from ..util.transform import dim
from .util import (
class GenericElementPlot(DimensionedPlot):
    """
    Plotting baseclass to render contents of an Element. Implements
    methods to get the correct frame given a HoloMap, axis labels and
    extents and titles.
    """
    apply_ranges = param.Boolean(default=True, doc='\n        Whether to compute the plot bounds from the data itself.')
    apply_extents = param.Boolean(default=True, doc='\n        Whether to apply extent overrides on the Elements')
    bgcolor = param.ClassSelector(class_=(str, tuple), default=None, doc='\n        If set bgcolor overrides the background color of the axis.')
    default_span = param.ClassSelector(default=2.0, class_=(int, float, tuple), doc='\n        Defines the span of an axis if the axis range is zero, i.e. if\n        the lower and upper end of an axis are equal or no range is\n        defined at all. For example if there is a single datapoint at\n        0 a default_span of 2.0 will result in axis ranges spanning\n        from -1 to 1.')
    hooks = param.HookList(default=[], doc='\n        Optional list of hooks called when finalizing a plot. The\n        hook is passed the plot object and the displayed element, and\n        other plotting handles can be accessed via plot.handles.')
    invert_axes = param.Boolean(default=False, doc='\n        Whether to invert the x- and y-axis')
    invert_xaxis = param.Boolean(default=False, doc='\n        Whether to invert the plot x-axis.')
    invert_yaxis = param.Boolean(default=False, doc='\n        Whether to invert the plot y-axis.')
    logx = param.Boolean(default=False, doc='\n        Whether the x-axis of the plot will be a log axis.')
    logy = param.Boolean(default=False, doc='\n        Whether the y-axis of the plot will be a log axis.')
    padding = param.ClassSelector(default=0.1, class_=(int, float, tuple), doc='\n        Fraction by which to increase auto-ranged extents to make\n        datapoints more visible around borders.\n\n        To compute padding, the axis whose screen size is largest is\n        chosen, and the range of that axis is increased by the\n        specified fraction along each axis.  Other axes are then\n        padded ensuring that the amount of screen space devoted to\n        padding is equal for all axes. If specified as a tuple, the\n        int or float values in the tuple will be used for padding in\n        each axis, in order (x,y or x,y,z).\n\n        For example, for padding=0.2 on a 800x800-pixel plot, an x-axis\n        with the range [0,10] will be padded by 20% to be [-1,11], while\n        a y-axis with a range [0,1000] will be padded to be [-100,1100],\n        which should make the padding be approximately the same number of\n        pixels. But if the same plot is changed to have a height of only\n        200, the y-range will then be [-400,1400] so that the y-axis\n        padding will still match that of the x-axis.\n\n        It is also possible to declare non-equal padding value for the\n        lower and upper bound of an axis by supplying nested tuples,\n        e.g. padding=(0.1, (0, 0.1)) will pad the x-axis lower and\n        upper bound as well as the y-axis upper bound by a fraction of\n        0.1 while the y-axis lower bound is not padded at all.')
    show_legend = param.Boolean(default=True, doc='\n        Whether to show legend for the plot.')
    show_grid = param.Boolean(default=False, doc='\n        Whether to show a Cartesian grid on the plot.')
    xaxis = param.ObjectSelector(default='bottom', objects=['top', 'bottom', 'bare', 'top-bare', 'bottom-bare', None, True, False], doc='\n        Whether and where to display the xaxis.\n        The "bare" options allow suppressing all axis labels, including ticks and xlabel.\n        Valid options are \'top\', \'bottom\', \'bare\', \'top-bare\' and \'bottom-bare\'.')
    yaxis = param.ObjectSelector(default='left', objects=['left', 'right', 'bare', 'left-bare', 'right-bare', None, True, False], doc='\n        Whether and where to display the yaxis.\n        The "bare" options allow suppressing all axis labels, including ticks and ylabel.\n        Valid options are \'left\', \'right\', \'bare\', \'left-bare\' and \'right-bare\'.')
    xlabel = param.String(default=None, doc='\n        An explicit override of the x-axis label, if set takes precedence\n        over the dimension label.')
    ylabel = param.String(default=None, doc='\n        An explicit override of the y-axis label, if set takes precedence\n        over the dimension label.')
    xlim = param.Tuple(default=(np.nan, np.nan), length=2, doc='\n       User-specified x-axis range limits for the plot, as a tuple (low,high).\n       If specified, takes precedence over data and dimension ranges.')
    ylim = param.Tuple(default=(np.nan, np.nan), length=2, doc='\n       User-specified x-axis range limits for the plot, as a tuple (low,high).\n       If specified, takes precedence over data and dimension ranges.')
    zlim = param.Tuple(default=(np.nan, np.nan), length=2, doc='\n       User-specified z-axis range limits for the plot, as a tuple (low,high).\n       If specified, takes precedence over data and dimension ranges.')
    xrotation = param.Integer(default=None, bounds=(0, 360), doc='\n        Rotation angle of the xticks.')
    yrotation = param.Integer(default=None, bounds=(0, 360), doc='\n        Rotation angle of the yticks.')
    xticks = param.Parameter(default=None, doc='\n        Ticks along x-axis specified as an integer, explicit list of\n        tick locations, or bokeh Ticker object. If set to None default\n        bokeh ticking behavior is applied.')
    yticks = param.Parameter(default=None, doc='\n        Ticks along y-axis specified as an integer, explicit list of\n        tick locations, or bokeh Ticker object. If set to None\n        default bokeh ticking behavior is applied.')
    _plot_methods = {}
    _propagate_options = []
    v17_option_propagation = True
    _deprecations = {'color_index': "The `color_index` parameter is deprecated in favor of color style mapping, e.g. `color=dim('color')` or `line_color=dim('color')`", 'size_index': "The `size_index` parameter is deprecated in favor of size style mapping, e.g. `size=dim('size')**2`.", 'scaling_method': "The `scaling_method` parameter is deprecated in favor of size style mapping, e.g. `size=dim('size')**2` for area scaling.", 'scaling_factor': "The `scaling_factor` parameter is deprecated in favor of size style mapping, e.g. `size=dim('size')*10`.", 'size_fn': "The `size_fn` parameter is deprecated in favor of size style mapping, e.g. `size=abs(dim('size'))`."}
    _selection_display = NoOpSelectionDisplay()
    _multi_y_propagation = False

    def __init__(self, element, keys=None, ranges=None, dimensions=None, batched=False, overlaid=0, cyclic_index=0, zorder=0, style=None, overlay_dims=None, stream_sources=None, streams=None, **params):
        if stream_sources is None:
            stream_sources = {}
        if overlay_dims is None:
            overlay_dims = {}
        self.zorder = zorder
        self.cyclic_index = cyclic_index
        self.overlaid = overlaid
        self.overlay_dims = overlay_dims
        if not isinstance(element, (HoloMap, DynamicMap)):
            self.hmap = HoloMap(initial_items=(0, element), kdims=['Frame'], id=element.id)
        else:
            self.hmap = element
        if overlaid:
            self.stream_sources = stream_sources
        else:
            self.stream_sources = compute_overlayable_zorders(self.hmap)
        plot_element = self.hmap.last
        if batched and (not isinstance(self, GenericOverlayPlot)):
            plot_element = plot_element.last
        dynamic = isinstance(element, DynamicMap) and (not element.unbounded)
        self.top_level = keys is None
        if self.top_level:
            dimensions = self.hmap.kdims
            keys = list(self.hmap.data.keys())
        self.style = self.lookup_options(plot_element, 'style') if style is None else style
        plot_opts = self.lookup_options(plot_element, 'plot').options
        propagate_options = self._propagate_options.copy()
        if self._multi_y_propagation:
            propagate_options = list(set(propagate_options) - set(GenericOverlayPlot._multi_y_unpropagated))
        if self.v17_option_propagation:
            inherited = self._traverse_options(plot_element, 'plot', propagate_options, defaults=False)
            plot_opts.update(**{k: v[0] for k, v in inherited.items() if k not in plot_opts})
        applied_params = dict(params, **plot_opts)
        for p, pval in applied_params.items():
            if p in self.param and p in self._deprecations and (pval is not None):
                self.param.warning(self._deprecations[p])
        super().__init__(keys=keys, dimensions=dimensions, dynamic=dynamic, **applied_params)
        self.batched = batched
        self.streams = get_nested_streams(self.hmap) if streams is None else streams
        if not (self.overlaid or (self.batched and (not isinstance(self, GenericOverlayPlot)))):
            attach_streams(self, self.hmap)
        if self.batched:
            self.ordering = util.layer_sort(self.hmap)
            overlay_opts = self.lookup_options(self.hmap.last, 'plot').options.items()
            opts = {k: v for k, v in overlay_opts if k in self.param}
            self.param.update(**opts)
            self.style = self.lookup_options(plot_element, 'style').max_cycles(len(self.ordering))
        else:
            self.ordering = []

    def get_zorder(self, overlay, key, el):
        """
        Computes the z-order of element in the NdOverlay
        taking into account possible batching of elements.
        """
        spec = util.get_overlay_spec(overlay, key, el)
        return self.ordering.index(spec)

    def _updated_zorders(self, overlay):
        specs = [util.get_overlay_spec(overlay, key, el) for key, el in overlay.data.items()]
        self.ordering = sorted(set(self.ordering + specs))
        return [self.ordering.index(spec) for spec in specs]

    def _get_axis_dims(self, element):
        """
        Returns the dimensions corresponding to each axis.

        Should return a list of dimensions or list of lists of
        dimensions, which will be formatted to label the axis
        and to link axes.
        """
        dims = element.dimensions()[:2]
        if len(dims) == 1:
            return dims + [None, None]
        else:
            return dims + [None]

    def _has_axis_dimension(self, element, dimension):
        dims = self._get_axis_dims(element)
        return any((dimension in ds if isinstance(ds, list) else dimension == ds for ds in dims))

    def _get_frame(self, key):
        if isinstance(self.hmap, DynamicMap) and self.overlaid and self.current_frame:
            self.current_key = key
            return self.current_frame
        elif key == self.current_key and (not self._force):
            return self.current_frame
        cached = self.current_key is None and (not any((s._triggering for s in self.streams)))
        key_map = dict(zip([d.name for d in self.dimensions], key))
        frame = get_plot_frame(self.hmap, key_map, cached)
        traverse_setter(self, '_force', False)
        if key not in self.keys and len(key) == self.hmap.ndims and self.dynamic:
            self.keys.append(key)
        self.current_frame = frame
        self.current_key = key
        return frame

    def _execute_hooks(self, element):
        """
        Executes finalize hooks
        """
        for hook in self.hooks:
            try:
                hook(self, element)
            except Exception as e:
                self.param.warning(f'Plotting hook {hook!r} could not be applied:\n\n {e}')

    def get_aspect(self, xspan, yspan):
        """
        Should define the aspect ratio of the plot.
        """

    def get_padding(self, obj, extents):
        """
        Computes padding along the axes taking into account the plot aspect.
        """
        x0, y0, z0, x1, y1, z1 = extents
        padding_opt = self.lookup_options(obj, 'plot').kwargs.get('padding')
        if self.overlaid:
            padding = 0
        elif padding_opt is None:
            if self.param.objects('existing')['padding'].default is not self.padding:
                padding = self.padding
            else:
                opts = self._traverse_options(obj, 'plot', ['padding'], specs=[Element], defaults=True)
                padding = opts.get('padding')
                if padding:
                    padding = padding[0]
                else:
                    padding = self.padding
        else:
            padding = padding_opt
        xpad, ypad, zpad = get_axis_padding(padding)
        if not self.overlaid and (not self.batched):
            xspan = x1 - x0 if util.is_number(x0) and util.is_number(x1) else None
            yspan = y1 - y0 if util.is_number(y0) and util.is_number(y1) else None
            aspect = self.get_aspect(xspan, yspan)
            if aspect > 1:
                xpad = tuple((xp / aspect for xp in xpad)) if isinstance(xpad, tuple) else xpad / aspect
            else:
                ypad = tuple((yp * aspect for yp in ypad)) if isinstance(ypad, tuple) else ypad * aspect
        return (xpad, ypad, zpad)

    def _get_range_extents(self, element, ranges, range_type, xdim, ydim, zdim):
        dims = element.dimensions()
        ndims = len(dims)
        xdim = xdim or (dims[0] if ndims else None)
        ydim = ydim or (dims[1] if ndims > 1 else None)
        if isinstance(self.projection, str) and self.projection == '3d':
            zdim = zdim or (dims[2] if ndims > 2 else None)
        else:
            zdim = None
        (x0, x1), xsrange, xhrange = get_range(element, ranges, xdim)
        (y0, y1), ysrange, yhrange = get_range(element, ranges, ydim)
        (z0, z1), zsrange, zhrange = get_range(element, ranges, zdim)
        trigger = False
        if not self.overlaid and (not self.batched):
            xspan, yspan, zspan = (v / 2.0 for v in get_axis_padding(self.default_span))
            mx0, mx1 = get_minimum_span(x0, x1, xspan)
            if x0 != mx0 or x1 != mx1:
                x0, x1 = (mx0, mx1)
                trigger = True
            my0, my1 = get_minimum_span(y0, y1, yspan)
            if y0 != my0 or y1 != my1:
                y0, y1 = (my0, my1)
                trigger = True
            mz0, mz1 = get_minimum_span(z0, z1, zspan)
        xpad, ypad, zpad = self.get_padding(element, (x0, y0, z0, x1, y1, z1))
        if range_type == 'soft':
            x0, x1 = xsrange
        elif range_type == 'hard':
            x0, x1 = xhrange
        elif xdim == 'categorical':
            x0, x1 = ('', '')
        elif range_type == 'combined':
            x0, x1 = util.dimension_range(x0, x1, xhrange, xsrange, xpad, self.logx)
        if range_type == 'soft':
            y0, y1 = ysrange
        elif range_type == 'hard':
            y0, y1 = yhrange
        elif range_type == 'combined':
            y0, y1 = util.dimension_range(y0, y1, yhrange, ysrange, ypad, self.logy)
        elif ydim == 'categorical':
            y0, y1 = ('', '')
        elif ydim is None:
            y0, y1 = (np.nan, np.nan)
        if isinstance(self.projection, str) and self.projection == '3d':
            if range_type == 'soft':
                z0, z1 = zsrange
            elif range_type == 'data':
                z0, z1 = zhrange
            elif range_type == 'combined':
                z0, z1 = util.dimension_range(z0, z1, zhrange, zsrange, zpad, self.logz)
            elif zdim == 'categorical':
                z0, z1 = ('', '')
            elif zdim is None:
                z0, z1 = (np.nan, np.nan)
            return (x0, y0, z0, x1, y1, z1)
        if not self.drawn:
            for stream in getattr(self, 'source_streams', []):
                if isinstance(stream, (RangeX, RangeY, RangeXY)) and trigger and (stream not in self._trigger):
                    self._trigger.append(stream)
        return (x0, y0, x1, y1)

    def get_extents(self, element, ranges, range_type='combined', dimension=None, xdim=None, ydim=None, zdim=None, **kwargs):
        """
        Gets the extents for the axes from the current Element. The globally
        computed ranges can optionally override the extents.

        The extents are computed by combining the data ranges, extents
        and dimension ranges. Each of these can be obtained individually
        by setting the range_type to one of:

        * 'data': Just the data ranges
        * 'extents': Element.extents
        * 'soft': Dimension.soft_range values
        * 'hard': Dimension.range values

        To obtain the combined range, which includes range padding the
        default may be used:

        * 'combined': All the range types combined and padding applied

        This allows Overlay plots to obtain each range and combine them
        appropriately for all the objects in the overlay.
        """
        num = 6 if isinstance(self.projection, str) and self.projection == '3d' else 4
        if self.apply_extents and range_type in ('combined', 'extents'):
            norm_opts = self.lookup_options(element, 'norm').options
            if norm_opts.get('framewise', False) or self.dynamic:
                extents = element.extents
            else:
                extent_list = self.hmap.traverse(lambda x: x.extents, [Element])
                extents = util.max_extents(extent_list, isinstance(self.projection, str) and self.projection == '3d')
        else:
            extents = (np.nan,) * num
        if range_type == 'extents':
            return extents
        if self.apply_ranges:
            range_extents = self._get_range_extents(element, ranges, range_type, xdim, ydim, zdim)
        else:
            range_extents = (np.nan,) * num
        if getattr(self, 'shared_axes', False) and self.subplot:
            combined = util.max_extents([range_extents, extents], isinstance(self.projection, str) and self.projection == '3d')
        else:
            max_extent = []
            for l1, l2 in zip(range_extents, extents):
                if isfinite(l2):
                    max_extent.append(l2)
                else:
                    max_extent.append(l1)
            combined = tuple(max_extent)
        if isinstance(self.projection, str) and self.projection == '3d':
            x0, y0, z0, x1, y1, z1 = combined
        else:
            x0, y0, x1, y1 = combined
        x0, x1 = util.dimension_range(x0, x1, self.xlim, (None, None))
        y0, y1 = util.dimension_range(y0, y1, self.ylim, (None, None))
        if not self.drawn:
            x_range, y_range = ((y0, y1), (x0, x1)) if self.invert_axes else ((x0, x1), (y0, y1))
            for stream in getattr(self, 'source_streams', []):
                if isinstance(stream, RangeX):
                    params = {'x_range': x_range}
                elif isinstance(stream, RangeY):
                    params = {'y_range': y_range}
                elif isinstance(stream, RangeXY):
                    params = {'x_range': x_range, 'y_range': y_range}
                else:
                    continue
                stream.update(**params)
                if stream not in self._trigger and (self.xlim or self.ylim):
                    self._trigger.append(stream)
        if isinstance(self.projection, str) and self.projection == '3d':
            z0, z1 = util.dimension_range(z0, z1, self.zlim, (None, None))
            return (x0, y0, z0, x1, y1, z1)
        return (x0, y0, x1, y1)

    def _get_axis_labels(self, dimensions, xlabel=None, ylabel=None, zlabel=None):
        if self.xlabel is not None:
            xlabel = self.xlabel
        elif dimensions and xlabel is None:
            xdims = dimensions[0]
            xlabel = dim_axis_label(xdims) if xdims else ''
        if self.ylabel is not None:
            ylabel = self.ylabel
        elif len(dimensions) >= 2 and ylabel is None:
            ydims = dimensions[1]
            ylabel = dim_axis_label(ydims) if ydims else ''
        if getattr(self, 'zlabel', None) is not None:
            zlabel = self.zlabel
        elif isinstance(self.projection, str) and self.projection == '3d' and (len(dimensions) >= 3) and (zlabel is None):
            zlabel = dim_axis_label(dimensions[2]) if dimensions[2] else ''
        return (xlabel, ylabel, zlabel)

    def _format_title_components(self, key, dimensions=True, separator='\n'):
        frame = self._get_frame(key)
        if frame is None:
            return ('', '', '', '')
        type_name = type(frame).__name__
        group = frame.group if frame.group != type_name else ''
        label = frame.label
        if self.layout_dimensions or dimensions:
            dim_title = self._frame_title(key, separator=separator)
        else:
            dim_title = ''
        return (label, group, type_name, dim_title)

    def _parse_backend_opt(self, opt, plot, model_accessor_aliases):
        """
        Parses a custom option of the form 'model.accessor.option'
        and returns the corresponding model and accessor.
        """
        accessors = opt.split('.')
        if len(accessors) < 2:
            self.param.warning(f"Custom option {opt!r} expects at least two accessors separated by '.'")
            return
        model_accessor = accessors[0]
        model_accessor = model_accessor_aliases.get(model_accessor) or model_accessor
        if model_accessor in self.handles:
            model = self.handles[model_accessor]
        elif hasattr(plot, model_accessor):
            model = getattr(plot, model_accessor)
        else:
            self.param.warning(f'{model_accessor} model could not be resolved on {type(self).__name__!r} plot. Ensure the {opt!r} custom option spec references a valid model in the plot.handles {list(self.handles.keys())!r} or on the underlying figure object.')
            return
        for acc in accessors[1:-1]:
            if '[' in acc and acc.endswith(']'):
                getitem_index = acc.index('[')
                getitem_spec = acc[getitem_index + 1:-1]
                try:
                    if ':' in getitem_spec:
                        slice_parts = getitem_spec.split(':')
                        slice_start = None if slice_parts[0] == '' else int(slice_parts[0])
                        slice_stop = None if slice_parts[1] == '' else int(slice_parts[1])
                        slice_step = None if len(slice_parts) < 3 or slice_parts[2] == '' else int(slice_parts[2])
                        getitem_acc = slice(slice_start, slice_stop, slice_step)
                    elif ',' in getitem_spec:
                        getitem_acc = [literal_eval(item.strip()) for item in getitem_spec.split(',')]
                    else:
                        getitem_acc = literal_eval(getitem_spec)
                except Exception:
                    self.param.warning(f'Could not evaluate getitem {getitem_spec!r} in custom option spec {opt!r}.')
                    model = None
                    break
                acc = acc[:getitem_index]
            else:
                getitem_acc = None
            if '(' in acc and ')' in acc:
                method_ini_index = acc.index('(')
                method_end_index = acc.index(')')
                method_spec = acc[method_ini_index + 1:method_end_index]
                try:
                    if method_spec:
                        method_parts = method_spec.split(',')
                        method_args = []
                        method_kwargs = {}
                        for part in method_parts:
                            if '=' in part:
                                key, value = part.split('=')
                                method_kwargs[key.strip()] = literal_eval(value.strip())
                            else:
                                method_args.append(literal_eval(part.strip()))
                    else:
                        method_args = ()
                        method_kwargs = {}
                except Exception:
                    self.param.warning(f'Could not evaluate method arguments {method_spec!r} in custom option spec {opt!r}.')
                    model = None
                    break
                acc = acc[:method_ini_index]
                if not isinstance(model, list):
                    model = getattr(model, acc)(*method_args, **method_kwargs)
                else:
                    model = [getattr(m, acc)(*method_args, **method_kwargs) for m in model]
                if getitem_acc is not None:
                    if not isinstance(getitem_acc, list):
                        model = model.__getitem__(getitem_acc)
                    else:
                        model = [model.__getitem__(i) for i in getitem_acc]
                acc = acc[method_end_index:]
            if acc == '' or model is None:
                continue
            if not hasattr(model, acc):
                self.param.warning(f'Could not resolve {acc!r} attribute on {type(model).__name__!r} model. Ensure the custom option spec you provided references a valid submodel.')
                model = None
                break
            model = getattr(model, acc)
        attr_accessor = accessors[-1]
        return (model, attr_accessor)

    def update_frame(self, key, ranges=None):
        """
        Set the plot(s) to the given frame number.  Operates by
        manipulating the matplotlib objects held in the self._handles
        dictionary.

        If n is greater than the number of available frames, update
        using the last available frame.
        """