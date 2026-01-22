class PatchMissing(BzrError):
    """Raise a patch type was specified but no patch supplied"""
    _fmt = 'Patch_type was %(patch_type)s, but no patch was supplied.'

    def __init__(self, patch_type):
        BzrError.__init__(self)
        self.patch_type = patch_type