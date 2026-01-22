class ForbiddenControlFileError(BzrError):
    _fmt = 'Cannot operate on "%(filename)s" because it is a control file'