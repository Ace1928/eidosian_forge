from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField
class NthValue(Func):
    function = 'NTH_VALUE'
    window_compatible = True

    def __init__(self, expression, nth=1, **extra):
        if expression is None:
            raise ValueError('%s requires a non-null source expression.' % self.__class__.__name__)
        if nth is None or nth <= 0:
            raise ValueError('%s requires a positive integer as for nth.' % self.__class__.__name__)
        super().__init__(expression, nth, **extra)

    def _resolve_output_field(self):
        sources = self.get_source_expressions()
        return sources[0].output_field