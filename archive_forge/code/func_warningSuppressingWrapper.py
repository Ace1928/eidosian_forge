import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
@wraps(f)
def warningSuppressingWrapper(*a, **kw):
    return runWithWarningsSuppressed(suppressedWarnings, f, *a, **kw)