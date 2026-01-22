class NoPublicBranch(BzrError):
    _fmt = 'There is no public branch set for "%(branch_url)s".'

    def __init__(self, branch):
        from . import urlutils
        public_location = urlutils.unescape_for_display(branch.base, 'ascii')
        BzrError.__init__(self, branch_url=public_location)