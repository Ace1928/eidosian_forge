import os
import warnings
class StreamModeError(ValueError):
    """Incorrect stream mode (text vs binary).

    This error should be raised when a stream (file or file-like object)
    argument is in text mode while the receiving function expects binary mode,
    or vice versa.
    """