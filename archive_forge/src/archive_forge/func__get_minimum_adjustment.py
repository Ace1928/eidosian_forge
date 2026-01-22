import math
def _get_minimum_adjustment(adjustment, min_adjustment_step):
    if min_adjustment_step and min_adjustment_step > abs(adjustment):
        adjustment = min_adjustment_step if adjustment > 0 else -min_adjustment_step
    return adjustment