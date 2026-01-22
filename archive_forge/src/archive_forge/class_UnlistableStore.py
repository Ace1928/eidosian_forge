class UnlistableStore(BzrError):

    def __init__(self, store):
        BzrError.__init__(self, 'Store %s is not listable' % store)