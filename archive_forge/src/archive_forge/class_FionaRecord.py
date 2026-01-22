import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
class FionaRecord(Record):
    """
    A single logical entry from a shapefile, combining the attributes with
    their associated geometry. This extends the standard Record to work
    with the FionaReader.

    """

    def __init__(self, geometry, attributes):
        self._geometry = geometry
        self.attributes = attributes
        if geometry is not None:
            self._bounds = geometry.bounds
        else:
            self._bounds = None