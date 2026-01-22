from fontTools.ttLib.tables import otTables as ot
from copy import deepcopy
import logging
def _limitFeatureVariationConditionRange(condition, axisLimit):
    minValue = condition.FilterRangeMinValue
    maxValue = condition.FilterRangeMaxValue
    if minValue > maxValue or minValue > axisLimit.maximum or maxValue < axisLimit.minimum:
        return
    return tuple((axisLimit.renormalizeValue(v, extrapolate=False) for v in (minValue, maxValue)))