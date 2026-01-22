import sys
from . import case
from . import util
class _DebugResult(object):
    """Used by the TestSuite to hold previous class when running in debug."""
    _previousTestClass = None
    _moduleSetUpFailed = False
    shouldStop = False