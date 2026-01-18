from django.contrib.gis.gdal.error import GDALException
def to_multi(self):
    """
        Transform Point, LineString, Polygon, and their 25D equivalents
        to their Multi... counterpart.
        """
    if self.name.startswith(('Point', 'LineString', 'Polygon')):
        self.num += 3