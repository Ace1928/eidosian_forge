import webob.dec
from osprofiler import _utils as utils
from osprofiler import profiler
def _trace_is_valid(self, trace_info):
    if not isinstance(trace_info, dict):
        return False
    trace_keys = set(trace_info.keys())
    if not all((k in trace_keys for k in _REQUIRED_KEYS)):
        return False
    if trace_keys.difference(_REQUIRED_KEYS + _OPTIONAL_KEYS):
        return False
    return True