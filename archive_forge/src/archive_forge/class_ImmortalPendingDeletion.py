class ImmortalPendingDeletion(BzrError):
    _fmt = 'Unable to delete transform temporary directory %(pending_deletion)s.  Please examine %(pending_deletion)s to see if it contains any files you wish to keep, and delete it when you are done.'

    def __init__(self, pending_deletion):
        BzrError.__init__(self, pending_deletion=pending_deletion)