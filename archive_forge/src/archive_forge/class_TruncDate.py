from datetime import datetime
from django.conf import settings
from django.db.models.expressions import Func
from django.db.models.fields import (
from django.db.models.lookups import (
from django.utils import timezone
class TruncDate(TruncBase):
    kind = 'date'
    lookup_name = 'date'
    output_field = DateField()

    def as_sql(self, compiler, connection):
        sql, params = compiler.compile(self.lhs)
        tzname = self.get_tzname()
        return connection.ops.datetime_cast_date_sql(sql, tuple(params), tzname)