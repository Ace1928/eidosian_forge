class UnstackableLocationError(BzrError):
    _fmt = "The branch '%(branch_url)s' cannot be stacked on '%(target_url)s'."

    def __init__(self, branch_url, target_url):
        BzrError.__init__(self)
        self.branch_url = branch_url
        self.target_url = target_url