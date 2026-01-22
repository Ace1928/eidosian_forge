from twisted.cred.error import UnauthorizedLogin
class UserRejectedKey(Exception):
    """
    The user interactively rejected a key.
    """