from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField
class LastValue(Func):
    arity = 1
    function = 'LAST_VALUE'
    window_compatible = True