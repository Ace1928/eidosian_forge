import signal
import weakref
from ... import trace
def unregister_on_hangup(identifier):
    """Remove a callback from being called during sighup."""
    if _on_sighup is None:
        return
    try:
        del _on_sighup[identifier]
    except KeyboardInterrupt:
        raise
    except Exception:
        trace.mutter('Error occurred during unregister_on_hangup:')
        trace.log_exception_quietly()