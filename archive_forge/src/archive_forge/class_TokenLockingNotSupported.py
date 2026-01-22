class TokenLockingNotSupported(LockError):
    _fmt = 'The object %(obj)s does not support token specifying a token when locking.'

    def __init__(self, obj):
        self.obj = obj