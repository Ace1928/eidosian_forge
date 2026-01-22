from django.contrib.gis.db.models.fields import (
from django.db.models import Aggregate, Func, Value
from django.utils.functional import cached_property
def as_oracle(self, compiler, connection, **extra_context):
    if not self.is_extent:
        tolerance = self.extra.get('tolerance') or getattr(self, 'tolerance', 0.05)
        clone = self.copy()
        source_expressions = self.get_source_expressions()
        if self.filter:
            source_expressions.pop()
        spatial_type_expr = Func(*source_expressions, Value(tolerance), function='SDOAGGRTYPE', output_field=self.output_field)
        source_expressions = [spatial_type_expr]
        if self.filter:
            source_expressions.append(self.filter)
        clone.set_source_expressions(source_expressions)
        return clone.as_sql(compiler, connection, **extra_context)
    return self.as_sql(compiler, connection, **extra_context)