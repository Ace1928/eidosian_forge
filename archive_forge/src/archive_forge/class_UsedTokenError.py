class UsedTokenError(TokenError):
    """
    Error raised by :mod:`passlib.totp` if a token is reused.
    Derives from :exc:`TokenError`.

    .. autoattribute:: expire_time

    .. versionadded:: 1.7
    """
    _default_message = 'Token has already been used, please wait for another.'
    expire_time = None

    def __init__(self, *args, **kwds):
        self.expire_time = kwds.pop('expire_time', None)
        TokenError.__init__(self, *args, **kwds)