import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
def _resetWarningFilters(passthrough, addedFilters):
    for f in addedFilters:
        try:
            warnings.filters.remove(f)
        except ValueError:
            pass
    return passthrough