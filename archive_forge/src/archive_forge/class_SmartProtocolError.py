class SmartProtocolError(TransportError):
    _fmt = 'Generic bzr smart protocol error: %(details)s'

    def __init__(self, details):
        self.details = details