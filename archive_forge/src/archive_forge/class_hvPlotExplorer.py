import holoviews as _hv
import numpy as np
import panel as pn
import param
from holoviews.core.util import datetime_types, dt_to_int, is_number, max_range
from holoviews.element import tile_sources
from holoviews.plotting.util import list_cmaps
from panel.viewable import Viewer
from .converter import HoloViewsConverter as _hvConverter
from .plotting import hvPlot as _hvPlot
from .util import is_geodataframe, is_xarray, instantiate_crs_str
class hvPlotExplorer(Viewer):
    kind = param.Selector()
    x = param.Selector()
    y = param.Selector()
    y_multi = param.ListSelector(default=[], label='Y')
    by = param.ListSelector(default=[])
    groupby = param.ListSelector(default=[])
    axes = param.ClassSelector(class_=Axes)
    colormapping = param.ClassSelector(class_=Colormapping)
    labels = param.ClassSelector(class_=Labels)
    geographic = param.ClassSelector(class_=Geographic)
    operations = param.ClassSelector(class_=Operations)
    statusbar = param.ClassSelector(class_=StatusBar)
    style = param.ClassSelector(class_=Style)
    advanced = param.ClassSelector(class_=Advanced)
    code = param.String(precedence=-1, doc='\n        Code to generate the plot.')

    @classmethod
    def from_data(cls, data, **params):
        if is_geodataframe(data):
            raise TypeError('GeoDataFrame objects not yet supported.')
        elif is_xarray(data):
            cls = hvGridExplorer
        else:
            cls = hvDataFrameExplorer
        return cls(data, **params)

    def __panel__(self):
        return self._layout

    def __init__(self, df, **params):
        x, y = (params.get('x'), params.get('y'))
        if 'y' in params:
            params['y_multi'] = params.pop('y') if isinstance(params['y'], list) else [params['y']]
        statusbar_params = {k: params.pop(k) for k in params.copy() if k in StatusBar.param}
        converter = _hvConverter(df, x, y, **{k: v for k, v in params.items() if k not in ('x', 'y', 'y_multi')})
        extras = {k: params.pop(k) for k in params.copy() if k not in self.param}
        super().__init__(**params)
        self._data = df
        self._converter = converter
        groups = {group: KINDS[group] for group in self._groups}
        self._controls = _create_param_pane(self, parameters=['kind', 'x', 'y', 'groupby', 'by'], widgets_kwargs={'kind': {'options': [], 'groups': groups}})
        self.param.watch(self._toggle_controls, 'kind')
        self.param.watch(self._check_y, 'y_multi')
        self.param.watch(self._check_by, 'by')
        self._populate()
        self._control_tabs = pn.Tabs(tabs_location='left')
        self.statusbar = StatusBar(**statusbar_params)
        self._statusbar = pn.Param(self.statusbar, show_name=False, default_layout=pn.Row, margin=(5, 56, 0, 56))
        controls = [p.class_ for p in self.param.objects().values() if isinstance(p, param.ClassSelector) and issubclass(p.class_, Controls)]
        controller_params = {}
        for cls in controls:
            controller_params[cls] = {k: extras.pop(k) for k in extras.copy() if k in cls.param}
        if extras:
            raise TypeError(f'__init__() got keyword(s) not supported by any control: {extras}')
        self._controllers = {cls.name.lower(): cls(df, explorer=self, **cparams) for cls, cparams in controller_params.items()}
        self.param.update(**self._controllers)
        self.param.watch(self._refresh, list(self.param))
        for controller in self._controllers.values():
            controller.param.watch(self._refresh, list(controller.param))
        self.statusbar.param.watch(self._refresh, list(self.statusbar.param))
        self._alert = pn.pane.Alert(alert_type='danger', visible=False, sizing_mode='stretch_width')
        self._hv_pane = pn.pane.HoloViews(sizing_mode='stretch_both', min_height=250, margin=(5, 5, 5, 20), widget_location='bottom', widget_layout=pn.Row)
        self._code_pane = pn.pane.Markdown(sizing_mode='stretch_both', min_height=250, margin=(5, 5, 0, 20))
        self._layout = pn.Column(self._alert, self._statusbar, pn.layout.Divider(), pn.Row(self._control_tabs, pn.Tabs(('Plot', self._hv_pane.layout), ('Code', self._code_pane)), sizing_mode='stretch_both'), sizing_mode='stretch_width', height=600)
        self.param.trigger('kind')

    def _populate(self):
        """
        Populates the options of the controls based on the data type.
        """
        variables = self._converter.variables
        indexes = getattr(self._converter, 'indexes', [])
        variables_no_index = [v for v in variables if v not in indexes]
        for pname in self.param:
            if pname == 'kind':
                continue
            p = self.param[pname]
            if isinstance(p, param.Selector):
                if pname == 'x':
                    p.objects = variables
                else:
                    p.objects = variables_no_index
                if (pname == 'x' or pname == 'y') and getattr(self, pname, None) is None:
                    setattr(self, pname, p.objects[0])

    def _plot(self):
        y = self.y_multi if 'y_multi' in self._controls.parameters else self.y
        if isinstance(y, list) and len(y) == 1:
            y = y[0]
        kwargs = {}
        for v in self.param.values().values():
            if isinstance(v, Geographic) and (not v.geo):
                continue
            if isinstance(v, Advanced):
                opts_kwargs = v.kwargs.get('opts', {})
            elif isinstance(v, Controls):
                kwargs.update(v.kwargs)
        if kwargs.get('geo'):
            if 'crs' not in kwargs:
                xmax = np.max(np.abs(self.xlim()))
                self.geographic.crs = 'PlateCarree' if xmax <= 360 else 'GOOGLE_MERCATOR'
                kwargs['crs'] = self.geographic.crs
            for key in ['crs', 'projection']:
                crs_kwargs = kwargs.pop(f'{key}_kwargs', {})
                kwargs[key] = instantiate_crs_str(kwargs.pop(key), **crs_kwargs)
            feature_scale = kwargs.pop('feature_scale', None)
            kwargs['features'] = {feature: feature_scale for feature in kwargs.pop('features', [])}
        kwargs['min_height'] = 400
        df = self._data
        if len(df) > MAX_ROWS and (not (self.kind in KINDS['stats'] or kwargs.get('rasterize') or kwargs.get('datashade'))):
            df = df.sample(n=MAX_ROWS)
        self._layout.loading = True
        try:
            self._hvplot = _hvPlot(df)(kind=self.kind, x=self.x, y=y, by=self.by, groupby=self.groupby, **kwargs)
            if opts_kwargs:
                self._hvplot.opts(**opts_kwargs)
            self._hv_pane.object = self._hvplot
            if len(self._hv_pane.widget_box) > 1:
                for w in self._hv_pane.widget_box:
                    w.margin = (20, 5, 5, 5)
            self._alert.visible = False
        except Exception as e:
            self._alert.param.update(object=f'**Rendering failed with following error**: {e}', visible=True)
        finally:
            self._layout.loading = False

    def _refresh(self, *events):
        if not self.statusbar.live_update:
            return
        self._plot()
        with param.parameterized.discard_events(self):
            self.code = self.plot_code()
        self._code_pane.object = f'```python\n{self.code}\n```'

    @property
    def _var_name(self):
        return 'data'

    @property
    def _single_y(self):
        if self.kind in KINDS['2d']:
            return True
        return False

    @property
    def _groups(self):
        raise NotImplementedError('Must be implemented by subclasses.')

    def _toggle_controls(self, event=None):
        visible = True
        if event and event.new in ('table', 'dataset'):
            parameters = ['kind', 'columns']
            visible = False
        elif event and event.new in KINDS['2d']:
            parameters = ['kind', 'x', 'y', 'by', 'groupby']
        elif event and event.new in ('hist', 'kde', 'density'):
            self.x = None
            parameters = ['kind', 'y_multi', 'by', 'groupby']
        else:
            parameters = ['kind', 'x', 'y_multi', 'by', 'groupby']
        self._controls.parameters = parameters
        tabs = [('Fields', self._controls)]
        if visible:
            tabs += [('Axes', self.axes), ('Labels', self.labels), ('Style', self.style), ('Operations', self.operations), ('Geographic', self.geographic), ('Advanced', self.advanced)]
            if event and event.new not in ('area', 'kde', 'line', 'ohlc', 'rgb', 'step'):
                tabs.insert(5, ('Colormapping', self.colormapping))
        self._control_tabs[:] = tabs

    def _check_y(self, event):
        if len(event.new) > 1 and self.by:
            self.y = event.old

    def _check_by(self, event):
        if event.new and 'y_multi' in self._controls.parameters and self.y_multi and (len(self.y_multi) > 1):
            self.by = []

    def hvplot(self):
        """Return the plot as a HoloViews object.
        """
        return self._hvplot.clone()

    def plot_code(self, var_name=None):
        """Return a string representation that can be easily copy-pasted
        in a notebook cell to create a plot from a call to the `.hvplot`
        accessor, and that includes all the customized settings of the explorer.

        >>> hvexplorer.plot_code(var_name='data')
        "data.hvplot(x='time', y='value')"

        Parameters
        ----------
        var_name: string
            Data variable name by which the returned string will start.
        """
        settings = self.settings()
        if 'legend' not in settings:
            settings['legend'] = 'bottom_right'
        settings['widget_location'] = 'bottom'
        settings_args = ''
        if settings:
            settings_args = self._build_kwargs_string(settings)
        snippet = f'{var_name or self._var_name}.hvplot(\n{settings_args}\n)'
        opts = self.advanced.opts
        if opts:
            opts_args = self._build_kwargs_string(opts)
            snippet += f'.opts(\n{opts_args}\n)'
        return snippet

    def _build_kwargs_string(self, kwargs):
        args = ''
        if kwargs:
            for k, v in kwargs.items():
                args += f'    {k}={v!r},\n'
            args = args[:-1]
        return args

    def save(self, filename, **kwargs):
        """Save the plot to file.

        Calls the `holoviews.save` utility, refer to its documentation
        for a full description of the available kwargs.

        Parameters
        ----------
        filename: string, pathlib.Path or IO object
            The path or BytesIO/StringIO object to save to
        """
        _hv.save(self._hvplot, filename, **kwargs)

    def settings(self):
        """Return a dictionary of the customized settings.

        This dictionary can be reused as an unpacked input to the explorer or
        a call to the `.hvplot` accessor.

        >>> hvplot.explorer(df, **settings)
        >>> df.hvplot(**settings)
        """
        settings = {}
        for controller in self._controllers.values():
            params = set(controller.param) - {'name', 'explorer'}
            for p in params:
                value = getattr(controller, p)
                if value != controller.param[p].default:
                    settings[p] = value
        for p in self._controls.parameters:
            value = getattr(self, p)
            if value != self.param[p].default or p == 'kind':
                settings[p] = value
        if 'y_multi' in settings:
            settings['y'] = settings.pop('y_multi')
        settings.pop('opts', None)
        settings = {k: v for k, v in sorted(list(settings.items()))}
        return settings