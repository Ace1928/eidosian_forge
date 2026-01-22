class TagsNotSupported(BzrError):
    _fmt = "Tags not supported by %(branch)s; you may be able to use 'brz upgrade %(branch_url)s'."

    def __init__(self, branch):
        self.branch = branch
        self.branch_url = branch.user_url