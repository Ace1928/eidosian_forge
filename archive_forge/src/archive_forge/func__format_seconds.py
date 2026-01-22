import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
def _format_seconds(self, sec):
    """Format seconds as an float"""
    return f'{sec:{self._seconds_num_format}}{self._second_symbol}'