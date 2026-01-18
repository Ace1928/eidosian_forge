import linecache
import re
def trace_filter(mode):
    """
    Set the trace filter mode.

    mode: Whether to enable the trace hook.
      True: Trace filtering on (skipping methods tagged @DontTrace)
      False: Trace filtering off (trace methods tagged @DontTrace)
      None/default: Toggle trace filtering.
    """
    global should_trace_hook
    if mode is None:
        mode = should_trace_hook is None
    if mode:
        should_trace_hook = default_should_trace_hook
    else:
        should_trace_hook = None
    return mode