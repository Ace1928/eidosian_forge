class PatchVerificationFailed(BzrError):
    """A patch from a merge directive could not be verified"""
    _fmt = 'Preview patch does not match requested changes.'