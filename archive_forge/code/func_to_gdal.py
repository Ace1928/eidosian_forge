from collections import namedtuple
import math
import warnings
def to_gdal(self):
    """Return same coefficient order as GDAL's SetGeoTransform().

        :rtype: tuple
        """
    return (self.c, self.a, self.b, self.f, self.d, self.e)