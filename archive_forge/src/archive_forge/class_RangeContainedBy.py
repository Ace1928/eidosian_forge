import datetime
import json
from django.contrib.postgres import forms, lookups
from django.db import models
from django.db.backends.postgresql.psycopg_any import (
from django.db.models.functions import Cast
from django.db.models.lookups import PostgresOperatorLookup
from .utils import AttributeSetter
class RangeContainedBy(PostgresOperatorLookup):
    lookup_name = 'contained_by'
    type_mapping = {'smallint': 'int4range', 'integer': 'int4range', 'bigint': 'int8range', 'double precision': 'numrange', 'numeric': 'numrange', 'date': 'daterange', 'timestamp with time zone': 'tstzrange'}
    postgres_operator = RangeOperators.CONTAINED_BY

    def process_rhs(self, compiler, connection):
        rhs, rhs_params = super().process_rhs(compiler, connection)
        db_type = self.lhs.output_field.cast_db_type(connection).split('(')[0]
        cast_type = self.type_mapping[db_type]
        return ('%s::%s' % (rhs, cast_type), rhs_params)

    def process_lhs(self, compiler, connection):
        lhs, lhs_params = super().process_lhs(compiler, connection)
        if isinstance(self.lhs.output_field, models.FloatField):
            lhs = '%s::numeric' % lhs
        elif isinstance(self.lhs.output_field, models.SmallIntegerField):
            lhs = '%s::integer' % lhs
        return (lhs, lhs_params)

    def get_prep_lookup(self):
        return RangeField().get_prep_value(self.rhs)