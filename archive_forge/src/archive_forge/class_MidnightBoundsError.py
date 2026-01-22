class MidnightBoundsError(RangeCheckError):
    """Raise when parsed time has an hour of 24 but is not midnight."""