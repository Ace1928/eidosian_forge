class ParseFormatError(BzrError):
    _fmt = 'Parse error on line %(lineno)d of %(format)s format: %(line)s'

    def __init__(self, format, lineno, line, text):
        BzrError.__init__(self)
        self.format = format
        self.lineno = lineno
        self.line = line
        self.text = text