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
class MeasureControl(Control):
    """MeasureControl class, with Control as parent class.

    A control which allows making measurements on the Map.

    Attributes
    ----------------------
    primary_length_unit: str, default 'feet'
        Possible values are 'feet', 'meters', 'miles', 'kilometers' or any user defined unit.
    secondary_length_unit: str, default None
        Possible values are 'feet', 'meters', 'miles', 'kilometers' or any user defined unit.
    primary_area_unit: str, default 'acres'
        Possible values are 'acres', 'hectares', 'sqfeet', 'sqmeters', 'sqmiles' or any user defined unit.
    secondary_area_unit: str, default None
        Possible values are 'acres', 'hectares', 'sqfeet', 'sqmeters', 'sqmiles' or any user defined unit.
    active_color: CSS Color, default '#ABE67E'
        The color used for current measurements.
    completed_color: CSS Color, default '#C8F2BE'
        The color used for the completed measurements.
    """
    _view_name = Unicode('LeafletMeasureControlView').tag(sync=True)
    _model_name = Unicode('LeafletMeasureControlModel').tag(sync=True)
    _length_units = ['feet', 'meters', 'miles', 'kilometers']
    _area_units = ['acres', 'hectares', 'sqfeet', 'sqmeters', 'sqmiles']
    _custom_units_dict = {}
    _custom_units = Dict().tag(sync=True)
    primary_length_unit = Enum(values=_length_units, default_value='feet', help='Possible values are feet, meters, miles, kilometers or any user\n                defined unit').tag(sync=True, o=True)
    secondary_length_unit = Enum(values=_length_units, default_value=None, allow_none=True, help='Possible values are feet, meters, miles, kilometers or any user\n                defined unit').tag(sync=True, o=True)
    primary_area_unit = Enum(values=_area_units, default_value='acres', help='Possible values are acres, hectares, sqfeet, sqmeters, sqmiles\n                or any user defined unit').tag(sync=True, o=True)
    secondary_area_unit = Enum(values=_area_units, default_value=None, allow_none=True, help='Possible values are acres, hectares, sqfeet, sqmeters, sqmiles\n                or any user defined unit').tag(sync=True, o=True)
    active_color = Color('#ABE67E').tag(sync=True, o=True)
    completed_color = Color('#C8F2BE').tag(sync=True, o=True)
    popup_options = Dict({'className': 'leaflet-measure-resultpopup', 'autoPanPadding': [10, 10]}).tag(sync=True, o=True)
    capture_z_index = Int(10000).tag(sync=True, o=True)

    def add_length_unit(self, name, factor, decimals=0):
        """Add a custom length unit.

        Parameters
        ----------
        name: str
            The name for your custom unit.
        factor: float
            Factor to apply when converting to this unit. Length in meters
            will be multiplied by this factor.
        decimals: int, default 0
            Number of decimals to round results when using this unit.
        """
        self._length_units.append(name)
        self._add_unit(name, factor, decimals)

    def add_area_unit(self, name, factor, decimals=0):
        """Add a custom area unit.

        Parameters
        ----------
        name: str
            The name for your custom unit.
        factor: float
            Factor to apply when converting to this unit. Area in sqmeters
            will be multiplied by this factor.
        decimals: int, default 0
            Number of decimals to round results when using this unit.
        """
        self._area_units.append(name)
        self._add_unit(name, factor, decimals)

    def _add_unit(self, name, factor, decimals):
        self._custom_units_dict[name] = {'factor': factor, 'display': name, 'decimals': decimals}
        self._custom_units = dict(**self._custom_units_dict)