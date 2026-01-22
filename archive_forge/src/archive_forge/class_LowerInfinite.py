import datetime
import json
from django.contrib.postgres import forms, lookups
from django.db import models
from django.db.backends.postgresql.psycopg_any import (
from django.db.models.functions import Cast
from django.db.models.lookups import PostgresOperatorLookup
from .utils import AttributeSetter
@RangeField.register_lookup
class LowerInfinite(models.Transform):
    lookup_name = 'lower_inf'
    function = 'LOWER_INF'
    output_field = models.BooleanField()