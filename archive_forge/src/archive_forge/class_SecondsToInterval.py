from django.db.models import DecimalField, DurationField, Func
class SecondsToInterval(Func):
    function = 'NUMTODSINTERVAL'
    template = "%(function)s(%(expressions)s, 'SECOND')"

    def __init__(self, expression, *, output_field=None, **extra):
        super().__init__(expression, output_field=output_field or DurationField(), **extra)