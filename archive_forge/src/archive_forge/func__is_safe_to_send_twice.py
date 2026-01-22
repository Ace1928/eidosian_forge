from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def _is_safe_to_send_twice(self):
    """Check if the current method is re-entrant safe."""
    if self.body_stream is not None or 'noretry' in debug.debug_flags:
        return False
    request_type = _mod_request.request_handlers.get_info(self.method)
    if request_type in ('read', 'idem', 'semi'):
        return True
    if request_type in ('semivfs', 'mutate', 'stream'):
        return False
    trace.mutter('Unknown request type: %s for method %s' % (request_type, self.method))
    return False