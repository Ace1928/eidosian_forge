class RepositoryDataStreamError(BzrError):
    _fmt = 'Corrupt or incompatible data stream: %(reason)s'

    def __init__(self, reason):
        self.reason = reason