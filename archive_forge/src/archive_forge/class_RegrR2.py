from django.db.models import Aggregate, FloatField, IntegerField
class RegrR2(StatAggregate):
    function = 'REGR_R2'