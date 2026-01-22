from django.contrib.gis.db.models.fields import (
from django.db.models import Aggregate, Func, Value
from django.utils.functional import cached_property
class MakeLine(GeoAggregate):
    name = 'MakeLine'
    output_field_class = LineStringField