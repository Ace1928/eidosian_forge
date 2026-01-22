class NoStatError(CloudPathException):
    """Used if stats cannot be retrieved; e.g., file does not exist
    or for some backends path is a directory (which doesn't have
    stats available).
    """