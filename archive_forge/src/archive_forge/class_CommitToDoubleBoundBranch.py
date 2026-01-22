class CommitToDoubleBoundBranch(BzrError):
    _fmt = 'Cannot commit to branch %(branch)s. It is bound to %(master)s, which is bound to %(remote)s.'

    def __init__(self, branch, master, remote):
        BzrError.__init__(self)
        self.branch = branch
        self.master = master
        self.remote = remote