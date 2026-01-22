from pathlib import Path
import sys
import inspect
import warnings
def _external_stacklevel(internal):
    """Find the stacklevel of the first frame that doesn't contain any of the given internal strings

    The depth will be 1 at minimum in order to start checking at the caller of
    the function that called this utility method.
    """
    level = 2
    frame = _get_frame(level)
    normalized_internal = [str(Path(s)) for s in internal]
    while frame and any((s in str(Path(frame.f_code.co_filename)) for s in normalized_internal)):
        level += 1
        frame = frame.f_back
    return level - 1