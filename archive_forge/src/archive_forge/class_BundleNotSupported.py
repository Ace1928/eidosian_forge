class BundleNotSupported(BzrError):
    _fmt = 'Unable to handle bundle version %(version)s: %(msg)s'

    def __init__(self, version, msg):
        BzrError.__init__(self)
        self.version = version
        self.msg = msg