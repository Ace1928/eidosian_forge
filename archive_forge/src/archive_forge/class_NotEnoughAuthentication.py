from twisted.cred.error import UnauthorizedLogin
class NotEnoughAuthentication(Exception):
    """
    This is thrown if the authentication is valid, but is not enough to
    successfully verify the user.  i.e. don't retry this type of
    authentication, try another one.
    """