class StrictCommitFailed(BzrError):
    _fmt = 'Commit refused because there are unknown files in the tree'