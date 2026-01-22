import warnings
import django
from django.db import models
from . import base
class SemVerField(models.CharField):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('max_length', 200)
        super(SemVerField, self).__init__(*args, **kwargs)

    def from_db_value(self, value, expression, connection, *args):
        """Convert from the database format.

        This should be the inverse of self.get_prep_value()
        """
        return self.to_python(value)

    def get_prep_value(self, obj):
        return None if obj is None else str(obj)

    def get_db_prep_value(self, value, connection, prepared=False):
        if not prepared:
            value = self.get_prep_value(value)
        return value

    def value_to_string(self, obj):
        value = self.to_python(self.value_from_object(obj))
        return str(value)

    def run_validators(self, value):
        return super(SemVerField, self).run_validators(str(value))