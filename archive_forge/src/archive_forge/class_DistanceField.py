from decimal import Decimal
from django.contrib.gis.measure import Area, Distance
from django.db import models
class DistanceField(models.FloatField):
    """Wrapper for Distance values."""

    def __init__(self, geo_field):
        super().__init__()
        self.geo_field = geo_field

    def get_prep_value(self, value):
        if isinstance(value, Distance):
            return value
        return super().get_prep_value(value)

    def get_db_prep_value(self, value, connection, prepared=False):
        if not isinstance(value, Distance):
            return value
        distance_att = connection.ops.get_distance_att_for_field(self.geo_field)
        if not distance_att:
            raise ValueError('Distance measure is supplied, but units are unknown for result.')
        return getattr(value, distance_att)

    def from_db_value(self, value, expression, connection):
        if value is None:
            return
        distance_att = connection.ops.get_distance_att_for_field(self.geo_field)
        return Distance(**{distance_att: value}) if distance_att else value

    def get_internal_type(self):
        return 'DistanceField'