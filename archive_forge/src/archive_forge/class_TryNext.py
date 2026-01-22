class TryNext(IPythonCoreError):
    """Try next hook exception.

    Raise this in your hook function to indicate that the next hook handler
    should be used to handle the operation.
    """