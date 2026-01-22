class StoringUncommittedNotSupported(BzrError):
    _fmt = 'Branch "%(display_url)s" does not support storing uncommitted changes.'

    def __init__(self, branch):
        import breezy.urlutils as urlutils
        user_url = getattr(branch, 'user_url', None)
        if user_url is None:
            display_url = str(branch)
        else:
            display_url = urlutils.unescape_for_display(user_url, 'ascii')
        BzrError.__init__(self, branch=branch, display_url=display_url)