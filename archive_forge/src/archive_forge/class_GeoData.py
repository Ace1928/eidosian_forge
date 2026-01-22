import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
class GeoData(GeoJSON):
    """GeoData class with GeoJSON as parent class.

    Layer created from a GeoPandas dataframe.

    Attributes
    ----------
    geo_dataframe: geopandas.GeoDataFrame instance, default None
        The GeoPandas dataframe to use.
    """
    geo_dataframe = Instance('geopandas.GeoDataFrame')

    def __init__(self, **kwargs):
        super(GeoData, self).__init__(**kwargs)
        self.data = self._get_data()

    @observe('geo_dataframe', 'style', 'style_callback')
    def _update_data(self, change):
        self.data = self._get_data()

    def _get_data(self):
        return json.loads(self.geo_dataframe.to_json())

    @property
    def __geo_interface__(self):
        """
        Return a dict whose structure aligns to the GeoJSON format
        For more information about the ``__geo_interface__``, see
        https://gist.github.com/sgillies/2217756
        """
        return self.geo_dataframe.__geo_interface__