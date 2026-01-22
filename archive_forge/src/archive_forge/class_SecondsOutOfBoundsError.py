class SecondsOutOfBoundsError(RangeCheckError):
    """Raise when parsed seconds are greater than 60."""