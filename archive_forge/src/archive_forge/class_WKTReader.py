from django.contrib.gis.geos.geometry import GEOSGeometry
from django.contrib.gis.geos.prototypes.io import (
class WKTReader(_WKTReader):

    def read(self, wkt):
        """Return a GEOSGeometry for the given WKT string."""
        return GEOSGeometry(super().read(wkt))