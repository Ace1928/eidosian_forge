import panel as _pn
import param
import holoviews as _hv
class hvplot_extension(_hv.extension):
    compatibility = param.ObjectSelector(allow_None=True, objects=['bokeh', 'matplotlib', 'plotly'], doc='\n            Plotting library used to process extra keyword arguments.')
    logo = param.Boolean(default=False)

    def __call__(self, *args, **params):
        compatibility = params.pop('compatibility', None)
        super().__call__(*args, **params)
        backend = _hv.Store.current_backend
        if compatibility in ['matplotlib', 'plotly'] and backend != compatibility:
            param.main.param.warning(f'Compatibility from {compatibility} to {backend} not yet implemented. Defaults to bokeh.')
        hvplot_extension.compatibility = compatibility
        from . import _patch_hvplot_docstrings
        _patch_hvplot_docstrings()