class NoWorkingTree(BzrError):
    _fmt = 'No WorkingTree exists for "%(base)s".'

    def __init__(self, base):
        BzrError.__init__(self)
        self.base = base