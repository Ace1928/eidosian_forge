from . import compat
class PasswordForbidden(ValidationError):
    """Exception raised when a URL has a password in the userinfo section."""

    def __init__(self, uri):
        """Initialize the error with the URI that failed validation."""
        unsplit = getattr(uri, 'unsplit', lambda: uri)
        super(PasswordForbidden, self).__init__('"{}" contained a password when validation forbade it'.format(unsplit()))
        self.uri = uri