class MissingBytes(ParsingError):
    """Raised when EOF encountered while expecting to find more bytes."""
    _fmt = _LOCATION_FMT + 'Unexpected EOF - expected %(expected)d bytes, found %(found)d'

    def __init__(self, lineno, expected, found):
        self.expected = expected
        self.found = found
        ParsingError.__init__(self, lineno)