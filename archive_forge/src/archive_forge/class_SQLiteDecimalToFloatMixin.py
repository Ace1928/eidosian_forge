from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
class SQLiteDecimalToFloatMixin:
    """
    By default, Decimal values are converted to str by the SQLite backend, which
    is not acceptable by the GIS functions expecting numeric values.
    """

    def as_sqlite(self, compiler, connection, **extra_context):
        copy = self.copy()
        copy.set_source_expressions([Value(float(expr.value)) if hasattr(expr, 'value') and isinstance(expr.value, Decimal) else expr for expr in copy.get_source_expressions()])
        return copy.as_sql(compiler, connection, **extra_context)