import os
import time
def _supports_progress(f):
    """Detect if we can use pretty progress bars on file F.

    If this returns true we expect that a human may be looking at that
    output, and that we can repaint a line to update it.

    This doesn't check the policy for whether we *should* use them.
    """
    isatty = getattr(f, 'isatty', None)
    if isatty is None:
        return False
    if not isatty():
        return False
    if os.environ.get('TERM') == 'dumb':
        return False
    return True