class UnknownParameter(ImportError):
    """Raised when an unknown parameter is passed to a processor."""
    _fmt = "Unknown parameter - '%(param)s' not in %(knowns)s"

    def __init__(self, param, knowns):
        self.param = param
        self.knowns = knowns
        ImportError.__init__(self)