class NoSubmitBranch(PathError):
    _fmt = 'No submit branch available for branch "%(path)s"'

    def __init__(self, branch):
        from . import urlutils
        self.path = urlutils.unescape_for_display(branch.base, 'ascii')