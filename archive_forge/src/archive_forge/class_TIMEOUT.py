import traceback
import sys
class TIMEOUT(ExceptionPexpect):
    """Raised when a read time exceeds the timeout. """