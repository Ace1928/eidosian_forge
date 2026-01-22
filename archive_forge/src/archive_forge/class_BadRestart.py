class BadRestart(ImportError):
    """Raised when the import stream and id-map do not match up."""
    _fmt = 'Bad restart - attempted to skip commit %(commit_id)s but matching revision-id is unknown'

    def __init__(self, commit_id):
        self.commit_id = commit_id
        ImportError.__init__(self)