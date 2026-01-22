class InvalidTimezone(ParsingError):
    """Raised when converting a string timezone to a seconds offset."""
    _fmt = _LOCATION_FMT + 'Timezone %(timezone)r could not be converted.%(reason)s'

    def __init__(self, lineno, timezone, reason=None):
        self.timezone = timezone
        if reason:
            self.reason = ' ' + reason
        else:
            self.reason = ''
        ParsingError.__init__(self, lineno)