class InvalidCommand(ParsingError):
    """Raised when an unknown command found."""
    _fmt = _LOCATION_FMT + "Invalid command '%(cmd)s'"

    def __init__(self, lineno, cmd):
        self.cmd = cmd
        ParsingError.__init__(self, lineno)