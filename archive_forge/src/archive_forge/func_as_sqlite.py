from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
def as_sqlite(self, compiler, connection, **extra_context):
    clone = self.copy()
    if len(self.source_expressions) < 4:
        clone.source_expressions.append(Value(0))
    return super(Translate, clone).as_sqlite(compiler, connection, **extra_context)