import re
from email import errors
from email._policybase import compat32
from collections import deque
from io import StringIO
def _pop_message(self):
    retval = self._msgstack.pop()
    if self._msgstack:
        self._cur = self._msgstack[-1]
    else:
        self._cur = None
    return retval