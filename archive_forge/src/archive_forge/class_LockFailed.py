class LockFailed(LockError):
    internal_error = False
    _fmt = 'Cannot lock %(lock)s: %(why)s'

    def __init__(self, lock, why):
        LockError.__init__(self, '')
        self.lock = lock
        self.why = why