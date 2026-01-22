class CorruptRepository(BzrError):
    _fmt = 'An error has been detected in the repository %(repo_path)s.\nPlease run brz reconcile on this repository.'

    def __init__(self, repo):
        BzrError.__init__(self)
        self.repo_path = repo.user_url