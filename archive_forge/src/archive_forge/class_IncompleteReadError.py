class IncompleteReadError(EOFError):
    """
    Incomplete read error. Attributes:

    - partial: read bytes string before the end of stream was reached
    - expected: total number of expected bytes (or None if unknown)
    """

    def __init__(self, partial, expected):
        r_expected = 'undefined' if expected is None else repr(expected)
        super().__init__(f'{len(partial)} bytes read on a total of {r_expected} expected bytes')
        self.partial = partial
        self.expected = expected

    def __reduce__(self):
        return (type(self), (self.partial, self.expected))