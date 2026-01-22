class UnlockableTransport(LockError):
    internal_error = False
    _fmt = 'Cannot lock: transport is read only: %(transport)s'

    def __init__(self, transport):
        self.transport = transport