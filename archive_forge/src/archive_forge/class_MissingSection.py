class MissingSection(ParsingError):
    """Raised when a section is required in a command but not present."""
    _fmt = _LOCATION_FMT + 'Command %(cmd)s is missing section %(section)s'

    def __init__(self, lineno, cmd, section):
        self.cmd = cmd
        self.section = section
        ParsingError.__init__(self, lineno)