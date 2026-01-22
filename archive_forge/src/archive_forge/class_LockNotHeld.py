class LockNotHeld(LockError):
    _fmt = 'Lock not held: %(lock)s'
    internal_error = False

    def __init__(self, lock):
        self.lock = lock