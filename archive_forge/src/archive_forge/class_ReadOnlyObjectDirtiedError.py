class ReadOnlyObjectDirtiedError(ReadOnlyError):
    _fmt = 'Cannot change object %(obj)r in read only transaction'

    def __init__(self, obj):
        self.obj = obj