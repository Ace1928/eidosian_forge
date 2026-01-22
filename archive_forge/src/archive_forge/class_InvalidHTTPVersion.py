class InvalidHTTPVersion(ParseException):

    def __init__(self, version):
        self.version = version

    def __str__(self):
        return 'Invalid HTTP Version: %r' % self.version