class PasslibConfigWarning(PasslibWarning):
    """Warning issued when non-fatal issue is found related to the configuration
    of a :class:`~passlib.context.CryptContext` instance.

    This occurs primarily in one of two cases:

    * The CryptContext contains rounds limits which exceed the hard limits
      imposed by the underlying algorithm.
    * An explicit rounds value was provided which exceeds the limits
      imposed by the CryptContext.

    In both of these cases, the code will perform correctly & securely;
    but the warning is issued as a sign the configuration may need updating.

    .. versionadded:: 1.6
    """