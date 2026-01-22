class BoundBranchOutOfDate(BzrError):
    _fmt = 'Bound branch %(branch)s is out of date with master branch %(master)s.%(extra_help)s'

    def __init__(self, branch, master):
        BzrError.__init__(self)
        self.branch = branch
        self.master = master
        self.extra_help = ''