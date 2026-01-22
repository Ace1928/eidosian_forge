from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
@BaseSpatialField.register_lookup
class IsValid(OracleToleranceMixin, GeoFuncMixin, Transform):
    lookup_name = 'isvalid'
    output_field = BooleanField()

    def as_oracle(self, compiler, connection, **extra_context):
        sql, params = super().as_oracle(compiler, connection, **extra_context)
        return ("CASE %s WHEN 'TRUE' THEN 1 ELSE 0 END" % sql, params)