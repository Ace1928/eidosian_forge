import os
def _fd(f):
    """Get a filedescriptor from something which could be a file or an fd."""
    return f.fileno() if hasattr(f, 'fileno') else f