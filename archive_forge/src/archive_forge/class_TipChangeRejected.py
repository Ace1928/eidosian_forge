class TipChangeRejected(BzrError):
    """A pre_change_branch_tip hook function may raise this to cleanly and
    explicitly abort a change to a branch tip.
    """
    _fmt = 'Tip change rejected: %(msg)s'

    def __init__(self, msg):
        self.msg = msg