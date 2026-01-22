class ObjectNotLocked(LockError):
    _fmt = '%(obj)r is not locked'

    def __init__(self, obj):
        self.obj = obj