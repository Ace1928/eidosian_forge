class InternalBackendError(RuntimeError):
    """
    Error raised if something unrecoverable goes wrong with backend call;
    such as if ``crypt.crypt()`` returning a malformed hash.

    .. versionadded:: 1.7.3
    """