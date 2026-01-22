class UnlistableBranch(BzrError):

    def __init__(self, br):
        BzrError.__init__(self, 'Stores for branch %s are not listable' % br)