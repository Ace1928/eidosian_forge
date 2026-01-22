import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
def _get_dms(self, x):
    """Convert to degrees, minutes, seconds

        Parameters
        ----------
        x: float or array of floats
            Degrees

        Return
        ------
        x: degrees rounded to the requested precision
        degs: degrees
        mins: minutes
        secs: seconds
        """
    self._precision = 6
    x = np.asarray(x, 'd')
    degs = np.round(x, self._precision).astype('i')
    y = (x - degs) * 60
    mins = np.round(y, self._precision).astype('i')
    secs = np.round((y - mins) * 60, self._precision - 3)
    return (x, degs, mins, secs)