class DegenerateDataWarning(RuntimeWarning):
    """Warns when data is degenerate and results may not be reliable."""

    def __init__(self, msg=None):
        if msg is None:
            msg = 'Degenerate data encountered; results may not be reliable.'
        self.args = (msg,)