class NotAMergeDirective(BzrError):
    """File starting with %(firstline)r is not a merge directive"""

    def __init__(self, firstline):
        BzrError.__init__(self, firstline=firstline)