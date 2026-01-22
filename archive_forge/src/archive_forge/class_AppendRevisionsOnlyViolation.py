class AppendRevisionsOnlyViolation(BzrError):
    _fmt = 'Operation denied because it would change the main history, which is not permitted by the append_revisions_only setting on branch "%(location)s".'

    def __init__(self, location):
        import breezy.urlutils as urlutils
        location = urlutils.unescape_for_display(location, 'ascii')
        BzrError.__init__(self, location=location)