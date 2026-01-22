from django.db.models import Aggregate, FloatField, IntegerField
class RegrIntercept(StatAggregate):
    function = 'REGR_INTERCEPT'