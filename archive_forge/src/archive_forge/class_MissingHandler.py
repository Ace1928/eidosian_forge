class MissingHandler(ImportError):
    """Raised when a processor can't handle a command."""
    _fmt = 'Missing handler for command %(cmd)s'

    def __init__(self, cmd):
        self.cmd = cmd
        ImportError.__init__(self)