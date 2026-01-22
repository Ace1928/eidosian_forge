class LockContention(LockError):
    _fmt = 'Could not acquire lock "%(lock)s": %(msg)s'
    internal_error = False

    def __init__(self, lock, msg=''):
        self.lock = lock
        self.msg = msg