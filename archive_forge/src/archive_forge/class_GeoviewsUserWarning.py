import inspect
import os
import warnings
import holoviews as hv
import param
from packaging.version import Version
class GeoviewsUserWarning(UserWarning):
    """A Geoviews-specific ``UserWarning`` subclass.
    Used to selectively filter Geoviews warnings for unconditional display.
    """