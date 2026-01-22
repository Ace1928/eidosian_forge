class BadSubsumeSource(BzrError):
    _fmt = "Can't subsume %(other_tree)s into %(tree)s. %(reason)s"

    def __init__(self, tree, other_tree, reason):
        self.tree = tree
        self.other_tree = other_tree
        self.reason = reason