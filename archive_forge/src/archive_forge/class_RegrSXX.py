from django.db.models import Aggregate, FloatField, IntegerField
class RegrSXX(StatAggregate):
    function = 'REGR_SXX'