import re
from email import errors
from email._policybase import compat32
from collections import deque
from io import StringIO
def _set_headersonly(self):
    self._headersonly = True