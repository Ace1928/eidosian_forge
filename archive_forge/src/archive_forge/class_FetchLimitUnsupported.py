class FetchLimitUnsupported(UnsupportedOperation):
    fmt = 'InterBranch %(interbranch)r does not support fetching limits.'

    def __init__(self, interbranch):
        BzrError.__init__(self, interbranch=interbranch)