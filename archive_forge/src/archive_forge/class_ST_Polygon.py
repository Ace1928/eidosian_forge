import re
from django.conf import settings
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.db.models import GeometryField, RasterField
from django.contrib.gis.gdal import GDALRaster
from django.contrib.gis.geos.geometry import GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.core.exceptions import ImproperlyConfigured
from django.db import NotSupportedError, ProgrammingError
from django.db.backends.postgresql.operations import DatabaseOperations
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from django.db.models import Func, Value
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
from .adapter import PostGISAdapter
from .models import PostGISGeometryColumns, PostGISSpatialRefSys
from .pgraster import from_pgraster
class ST_Polygon(Func):
    function = 'ST_Polygon'

    def __init__(self, expr):
        super().__init__(expr)
        expr = self.source_expressions[0]
        if isinstance(expr, Value) and (not expr._output_field_or_none):
            self.source_expressions[0] = Value(expr.value, output_field=RasterField(srid=expr.value.srid))

    @cached_property
    def output_field(self):
        return GeometryField(srid=self.source_expressions[0].field.srid)