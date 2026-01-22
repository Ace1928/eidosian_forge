class UnexpectedHttpStatus(InvalidHttpResponse):
    _fmt = 'Unexpected HTTP status %(code)d for %(path)s: %(extra)s'

    def __init__(self, path, code, extra=None, headers=None):
        self.path = path
        self.code = code
        self.extra = extra or ''
        full_msg = 'status code %d unexpected' % code
        if extra is not None:
            full_msg += ': ' + extra
        InvalidHttpResponse.__init__(self, path, full_msg, headers=headers)