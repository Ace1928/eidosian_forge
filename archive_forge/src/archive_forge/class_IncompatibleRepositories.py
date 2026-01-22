class IncompatibleRepositories(BzrError):
    """Report an error that two repositories are not compatible.

    Note that the source and target repositories are permitted to be strings:
    this exception is thrown from the smart server and may refer to a
    repository the client hasn't opened.
    """
    _fmt = '%(target)s\nis not compatible with\n%(source)s\n%(details)s'

    def __init__(self, source, target, details=None):
        if details is None:
            details = '(no details)'
        BzrError.__init__(self, target=target, source=source, details=details)