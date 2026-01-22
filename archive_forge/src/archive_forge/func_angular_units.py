from django.contrib.gis import gdal
@property
def angular_units(self):
    """Return the angular units."""
    return self.srs.angular_units