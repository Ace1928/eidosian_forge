import weakref
from pydispatch import saferef, robustapply, errors
def allReceivers():
    for signal, set in items:
        for item in set:
            yield item