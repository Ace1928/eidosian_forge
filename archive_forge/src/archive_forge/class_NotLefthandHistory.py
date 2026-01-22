class NotLefthandHistory(InternalBzrError):
    _fmt = 'Supplied history does not follow left-hand parents'

    def __init__(self, history):
        BzrError.__init__(self, history=history)