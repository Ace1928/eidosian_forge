import itertools
from collections import defaultdict
import param
from ..converter import HoloViewsConverter
from ..util import is_list_like, process_dynamic_args
class hvPlotBase:
    """
    Internal base class.

    Concrete subclasses must implement plotting methods (e.g. `line`, `scatter`, `image`).
    A plotting method must call `self` which will effectively create a HoloViewsConverter
    and call it to return a HoloViews object.

    Concrete subclasses are meant to be mounted onto a datastructure property, e.g.:

    ```
    _patch_plot = lambda self: hvPlotTabular(self)
    _patch_plot.__doc__ = hvPlotTabular.__call__.__doc__
    plot_prop = property(_patch_plot)
    setattr(pd.DataFrame, 'hvplot', plot_prop)
    ```
    """
    __all__ = []

    def __init__(self, data, custom_plots={}, **metadata):
        if 'query' in metadata:
            data = data.query(metadata.pop('query'))
        if 'sel' in metadata:
            data = data.sel(**metadata.pop('sel'))
        if 'isel' in metadata:
            data = data.isel(**metadata.pop('isel'))
        self._data = data
        self._plots = custom_plots
        self._metadata = metadata

    def __call__(self, x=None, y=None, kind=None, **kwds):
        x = list(x) if is_list_like(x) else x
        y = list(y) if is_list_like(y) else y
        if isinstance(kind, str) and kind not in self.__all__:
            raise NotImplementedError(f"kind='{kind}' for data of type {type(self._data)}")
        if isinstance(kind, str) and kind == 'explorer':
            return self.explorer(x=x, y=y, **kwds)
        if panel_available:
            panel_args = ['widgets', 'widget_location', 'widget_layout', 'widget_type']
            panel_dict = {}
            for k in panel_args:
                if k in kwds:
                    panel_dict[k] = kwds.pop(k)
            dynamic, arg_deps, arg_names = process_dynamic_args(x, y, kind, **kwds)
            if dynamic or arg_deps:

                @pn.depends(*arg_deps, **dynamic)
                def callback(*args, **dyn_kwds):
                    xd = dyn_kwds.pop('x', x)
                    yd = dyn_kwds.pop('y', y)
                    kindd = dyn_kwds.pop('kind', kind)
                    combined_kwds = dict(kwds, **dyn_kwds)
                    fn_args = defaultdict(list)
                    for name, arg in zip(arg_names, args):
                        fn_args[name, kwds[name]].append(arg)
                    for (name, fn), args in fn_args.items():
                        combined_kwds[name] = fn(*args)
                    plot = self._get_converter(xd, yd, kindd, **combined_kwds)(kindd, xd, yd)
                    return pn.panel(plot, **panel_dict)
                return pn.panel(callback)
            if panel_dict:
                plot = self._get_converter(x, y, kind, **kwds)(kind, x, y)
                return pn.panel(plot, **panel_dict)
        return self._get_converter(x, y, kind, **kwds)(kind, x, y)

    def _get_converter(self, x=None, y=None, kind=None, **kwds):
        params = dict(self._metadata, **kwds)
        x = x or params.pop('x', None)
        y = y or params.pop('y', None)
        kind = kind or params.pop('kind', None)
        return HoloViewsConverter(self._data, x, y, kind=kind, **params)

    def __dir__(self):
        """
        List default attributes and custom defined plots.
        """
        dirs = super().__dir__()
        return sorted(list(dirs) + list(self._plots))

    def __getattribute__(self, name):
        """
        Custom getattribute to expose user defined subplots.
        """
        plots = object.__getattribute__(self, '_plots')
        if name in plots:
            plot_opts = plots[name]
            if 'kind' in plot_opts and name in HoloViewsConverter._kind_mapping:
                param.main.param.warning("Custom options for existing plot types should not declare the 'kind' argument. The .{} plot method was unexpectedly customized with kind={!r}.".format(plot_opts['kind'], name))
                plot_opts['kind'] = name
            return hvPlotBase(self._data, **dict(self._metadata, **plot_opts))
        return super().__getattribute__(name)

    def explorer(self, x=None, y=None, **kwds):
        """
        The `explorer` plot allows you to interactively explore your data.

        Reference: https://hvplot.holoviz.org/user_guide/Explorer.html

        Parameters
        ----------
        x : string, optional
            The coordinate variable along the x-axis
        y : string, optional
            The coordinate variable along the y-axis
        **kwds : optional
            Additional keywords arguments typically passed to hvplot's call.

        Returns
        -------
        The corresponding explorer type based on data, e.g. hvplot.ui.hvDataFrameExplorer.

        Examples
        --------

        .. code-block:

            import hvplot.pandas
            import pandas as pd

            df = pd.DataFrame(
                {
                    "actual": [100, 150, 125, 140, 145, 135, 123],
                    "forecast": [90, 160, 125, 150, 141, 141, 120],
                    "numerical": [1.1, 1.9, 3.2, 3.8, 4.3, 5.0, 5.5],
                    "date": pd.date_range("2022-01-03", "2022-01-09"),
                    "string": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                },
            )
            df.hvplot.explorer()
        """
        from ..ui import explorer as ui_explorer
        return ui_explorer(self._data, x=x, y=y, **kwds)