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
class Geographic(Controls):
    tiles = param.ObjectSelector(default=None, objects=GEO_TILES, doc="\n        Whether to overlay the plot on a tile source. Tiles sources\n        can be selected by name or a tiles object or class can be passed,\n        the default is 'Wikipedia'.")
    geo = param.Boolean(default=False, doc='\n        Whether the plot should be treated as geographic (and assume\n        PlateCarree, i.e. lat/lon coordinates). Require GeoViews.')
    crs = param.Selector(default=None, doc='\n        Coordinate reference system of the data specified as Cartopy\n        CRS object, proj.4 string or EPSG code.')
    crs_kwargs = param.Dict(default={}, doc='\n        Keyword arguments to pass to selected CRS.')
    projection = param.ObjectSelector(default=None, doc='\n        Projection to use for cartographic plots.')
    projection_kwargs = param.Dict(default={}, doc='\n        Keyword arguments to pass to selected projection.')
    global_extent = param.Boolean(default=None, doc='\n        Whether to expand the plot extent to span the whole globe.')
    project = param.Boolean(default=False, doc='\n        Whether to project the data before plotting (adds initial\n        overhead but avoids projecting data when plot is dynamically\n        updated).')
    features = param.ListSelector(default=None, objects=GEO_FEATURES, doc="\n        A list of features or a dictionary of features and the scale\n        at which to render it. Available features include 'borders',\n        'coastline', 'lakes', 'land', 'ocean', 'rivers' and 'states'.")
    feature_scale = param.ObjectSelector(default='110m', objects=['110m', '50m', '10m'], doc='\n        The scale at which to render the features.')
    _widgets_kwargs = {'geo': {'type': pn.widgets.Toggle}}

    def __init__(self, data, **params):
        gv_available = False
        try:
            import geoviews
            gv_available = True
        except ImportError:
            pass
        geo_params = GEO_KEYS + ['geo']
        if not gv_available and any((p in params for p in geo_params)):
            raise ImportError('GeoViews must be installed to enable the geographic options.')
        super().__init__(data, **params)
        if not gv_available:
            for p in geo_params:
                self.param[p].constant = True
            self.param['geo'].label = 'geo (require GeoViews)'
        else:
            self._update_crs_projection()

    @param.depends('geo', watch=True)
    def _update_crs_projection(self):
        enabled = bool(self.geo or self.project)
        for key in GEO_KEYS:
            self.param[key].constant = not enabled
        self.geo = enabled
        if not enabled:
            return
        from cartopy.crs import CRS
        crs_list = sorted((k for k in param.concrete_descendents(CRS).keys() if not k.startswith('_') and k != 'CRS'))
        crs_list.insert(0, 'GOOGLE_MERCATOR')
        crs_list.insert(0, 'PlateCarree')
        crs_list.remove('PlateCarree')
        self.param.crs.objects = crs_list
        self.param.projection.objects = crs_list
        updates = {}
        if self.projection is None:
            updates['projection'] = crs_list[0]
        if self.global_extent is None:
            updates['global_extent'] = True
        if self.features is None:
            updates['features'] = ['coastline']
        self.param.update(**updates)