from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField
class CumeDist(Func):
    function = 'CUME_DIST'
    output_field = FloatField()
    window_compatible = True