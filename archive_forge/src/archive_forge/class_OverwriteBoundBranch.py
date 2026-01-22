class OverwriteBoundBranch(BzrError):
    _fmt = 'Cannot pull --overwrite to a branch which is bound %(branch)s'

    def __init__(self, branch):
        BzrError.__init__(self)
        self.branch = branch