class UnknownHashError(ValueError):
    """
    Error raised by :class:`~passlib.crypto.lookup_hash` if hash name is not recognized.
    This exception derives from :exc:`!ValueError`.

    As of version 1.7.3, this may also be raised if hash algorithm is known,
    but has been disabled due to FIPS mode (message will include phrase "disabled for fips").

    As of version 1.7.4, this may be raised if a :class:`~passlib.context.CryptContext`
    is unable to identify the algorithm used by a password hash.

    .. versionadded:: 1.7

    .. versionchanged: 1.7.3
        added 'message' argument.

    .. versionchanged:: 1.7.4
        altered call signature.
    """

    def __init__(self, message=None, value=None):
        self.value = value
        if message is None:
            message = 'unknown hash algorithm: %r' % value
        self.message = message
        ValueError.__init__(self, message, value)

    def __str__(self):
        return self.message