import datetime
import json
from django.contrib.postgres import forms, lookups
from django.db import models
from django.db.backends.postgresql.psycopg_any import (
from django.db.models.functions import Cast
from django.db.models.lookups import PostgresOperatorLookup
from .utils import AttributeSetter
def as_postgresql(self, compiler, connection):
    sql, params = super().as_postgresql(compiler, connection)
    cast_sql = ''
    if isinstance(self.rhs, models.Expression) and self.rhs._output_field_or_none and (not isinstance(self.rhs._output_field_or_none, self.lhs.output_field.__class__)):
        cast_internal_type = self.lhs.output_field.base_field.get_internal_type()
        cast_sql = '::{}'.format(connection.data_types.get(cast_internal_type))
    return ('%s%s' % (sql, cast_sql), params)