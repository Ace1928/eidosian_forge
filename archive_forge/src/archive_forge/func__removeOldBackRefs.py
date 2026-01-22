import weakref
from pydispatch import saferef, robustapply, errors
def _removeOldBackRefs(senderkey, signal, receiver, receivers):
    """Kill old sendersBack references from receiver

    This guards against multiple registration of the same
    receiver for a given signal and sender leaking memory
    as old back reference records build up.

    Also removes old receiver instance from receivers
    """
    try:
        index = receivers.index(receiver)
    except ValueError:
        return False
    else:
        oldReceiver = receivers[index]
        del receivers[index]
        found = 0
        signals = connections.get(signal)
        if signals is not None:
            for sig, recs in connections.get(signal, {}).items():
                if sig != signal:
                    for rec in recs:
                        if rec is oldReceiver:
                            found = 1
                            break
        if not found:
            _killBackref(oldReceiver, senderkey)
            return True
        return False