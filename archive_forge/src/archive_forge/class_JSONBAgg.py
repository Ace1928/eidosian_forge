import json
import warnings
from django.contrib.postgres.fields import ArrayField
from django.db.models import Aggregate, BooleanField, JSONField, TextField, Value
from django.utils.deprecation import RemovedInDjango51Warning
from .mixins import OrderableAggMixin
class JSONBAgg(OrderableAggMixin, Aggregate):
    function = 'JSONB_AGG'
    template = '%(function)s(%(distinct)s%(expressions)s %(ordering)s)'
    allow_distinct = True
    output_field = JSONField()

    def __init__(self, *expressions, default=None, **extra):
        super().__init__(*expressions, default=default, **extra)
        if isinstance(default, Value) and isinstance(default.value, str) and (not isinstance(default.output_field, JSONField)):
            value = default.value
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                warnings.warn(f"Passing a Value() with an output_field that isn't a JSONField as JSONBAgg(default) is deprecated. Pass default=Value({value!r}, output_field=JSONField()) instead.", stacklevel=2, category=RemovedInDjango51Warning)
                self.default.output_field = self.output_field
            else:
                self.default = Value(decoded, self.output_field)
                warnings.warn(f'Passing an encoded JSON string as JSONBAgg(default) is deprecated. Pass default={decoded!r} instead.', stacklevel=2, category=RemovedInDjango51Warning)