class TransportError(BzrError):
    _fmt = 'Transport error: %(msg)s %(orig_error)s'

    def __init__(self, msg=None, orig_error=None):
        if msg is None and orig_error is not None:
            msg = str(orig_error)
        if orig_error is None:
            orig_error = ''
        if msg is None:
            msg = ''
        self.msg = msg
        self.orig_error = orig_error
        BzrError.__init__(self)