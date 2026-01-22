import sys
from . import case
from . import util
def _get_previous_module(self, result):
    previousModule = None
    previousClass = getattr(result, '_previousTestClass', None)
    if previousClass is not None:
        previousModule = previousClass.__module__
    return previousModule