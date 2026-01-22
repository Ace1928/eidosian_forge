class PasslibRuntimeWarning(PasslibWarning):
    """Warning issued when something unexpected happens during runtime.

    The fact that it's a warning instead of an error means Passlib
    was able to correct for the issue, but that it's anomalous enough
    that the developers would love to hear under what conditions it occurred.

    .. versionadded:: 1.6
    """