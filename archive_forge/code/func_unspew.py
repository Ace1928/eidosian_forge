import sys
import linecache
import re
import inspect
def unspew():
    """Remove the trace hook installed by spew.
    """
    sys.settrace(None)