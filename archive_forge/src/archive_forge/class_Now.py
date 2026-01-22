from datetime import datetime
from django.conf import settings
from django.db.models.expressions import Func
from django.db.models.fields import (
from django.db.models.lookups import (
from django.utils import timezone
class Now(Func):
    template = 'CURRENT_TIMESTAMP'
    output_field = DateTimeField()

    def as_postgresql(self, compiler, connection, **extra_context):
        return self.as_sql(compiler, connection, template='STATEMENT_TIMESTAMP()', **extra_context)

    def as_mysql(self, compiler, connection, **extra_context):
        return self.as_sql(compiler, connection, template='CURRENT_TIMESTAMP(6)', **extra_context)

    def as_sqlite(self, compiler, connection, **extra_context):
        return self.as_sql(compiler, connection, template="STRFTIME('%%%%Y-%%%%m-%%%%d %%%%H:%%%%M:%%%%f', 'NOW')", **extra_context)

    def as_oracle(self, compiler, connection, **extra_context):
        return self.as_sql(compiler, connection, template='LOCALTIMESTAMP', **extra_context)