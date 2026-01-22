import signal
import weakref
from ... import trace
def _setup_on_hangup_dict():
    """Create something for _on_sighup.

    This is done when we install the sighup handler, and for tests that want to
    test the functionality. If this hasn'nt been called, then
    register_on_hangup is a no-op. As is unregister_on_hangup.
    """
    global _on_sighup
    old = _on_sighup
    _on_sighup = weakref.WeakValueDictionary()
    return old