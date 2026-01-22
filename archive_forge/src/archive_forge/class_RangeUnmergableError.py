class RangeUnmergableError(ValueError):
    """Exception class when byte ranges are noncontiguous and can not be merged together."""

    def __init__(self, reason=None):
        if not reason:
            reason = 'Ranges can not be merged together'
        ValueError.__init__(self, reason)