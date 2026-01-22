from .errors import BzrError, InternalBzrError
class InvalidImportLine(InternalBzrError):
    _fmt = 'Not a valid import statement: %(msg)\n%(text)s'

    def __init__(self, text, msg):
        BzrError.__init__(self)
        self.text = text
        self.msg = msg