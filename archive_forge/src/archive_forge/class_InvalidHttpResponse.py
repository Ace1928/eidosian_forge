class InvalidHttpResponse(TransportError):
    _fmt = 'Invalid http response for %(path)s: %(msg)s%(orig_error)s'

    def __init__(self, path, msg, orig_error=None, headers=None):
        self.path = path
        if orig_error is None:
            orig_error = ''
        else:
            orig_error = ': {!r}'.format(orig_error)
        self.headers = headers
        TransportError.__init__(self, msg, orig_error=orig_error)