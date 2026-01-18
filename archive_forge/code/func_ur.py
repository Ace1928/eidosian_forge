from ctypes import Structure, c_double
from django.contrib.gis.gdal.error import GDALException
@property
def ur(self):
    """Return the upper-right coordinate."""
    return (self.max_x, self.max_y)