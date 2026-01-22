import bisect
def _distance_from_interval(point, interval, tolerance=None):
    lo, hi = interval
    if tolerance is None:
        tolerance = 0.0
    if point < lo - tolerance:
        return lo - point
    elif lo - tolerance <= point and point <= hi + tolerance:
        return 0.0
    elif point > hi + tolerance:
        return point - hi