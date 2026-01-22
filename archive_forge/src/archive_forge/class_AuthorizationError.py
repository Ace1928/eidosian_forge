import hashlib
class AuthorizationError(CommError):
    """The signature was valid, but the user is not permitted to do the
    requested action."""
    pass