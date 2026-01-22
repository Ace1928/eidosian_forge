class PasslibSecurityError(RuntimeError):
    """
    Error raised if critical security issue is detected
    (e.g. an attempt is made to use a vulnerable version of a bcrypt backend).

    .. versionadded:: 1.6.3
    """