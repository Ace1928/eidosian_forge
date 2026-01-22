class SubsumeTargetNeedsUpgrade(BzrError):
    _fmt = 'Subsume target %(other_tree)s needs to be upgraded.'

    def __init__(self, other_tree):
        self.other_tree = other_tree