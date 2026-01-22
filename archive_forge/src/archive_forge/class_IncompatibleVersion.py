class IncompatibleVersion(BzrError):
    _fmt = 'API %(api)s is not compatible; one of versions %(wanted)r is required, but current version is %(current)r.'

    def __init__(self, api, wanted, current):
        self.api = api
        self.wanted = wanted
        self.current = current