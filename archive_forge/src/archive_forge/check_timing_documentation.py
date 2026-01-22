from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
 Return string containing a benchmark report.

    The arguments are the name of the benchmark case, the times to do a 'get'
    or a 'set' operation for that benchmark case in usec, and the
    corresponding times for a reference operation (e.g., getting and
    setting an attribute on a new-style instance.
    