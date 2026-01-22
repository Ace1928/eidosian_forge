class InvalidHttpContentType(InvalidHttpResponse):
    _fmt = 'Invalid http Content-type "%(ctype)s" for %(path)s: %(msg)s'

    def __init__(self, path, ctype, msg):
        self.ctype = ctype
        InvalidHttpResponse.__init__(self, path, msg)