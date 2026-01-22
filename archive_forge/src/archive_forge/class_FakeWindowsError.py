import os
import re
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
class FakeWindowsError(OSError):
    """
    Stand-in for sometimes-builtin exception on platforms for which it
    is missing.
    """