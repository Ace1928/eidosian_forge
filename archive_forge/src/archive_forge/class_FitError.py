class FitError(RuntimeError):
    """Represents an error condition when fitting a distribution to data."""

    def __init__(self, msg=None):
        if msg is None:
            msg = 'An error occurred when fitting a distribution to data.'
        self.args = (msg,)