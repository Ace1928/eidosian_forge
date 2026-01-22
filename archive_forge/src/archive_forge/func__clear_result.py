import re
import warnings
from . import err
def _clear_result(self):
    self.rownumber = 0
    self._result = None
    self.rowcount = 0
    self.warning_count = 0
    self.description = None
    self.lastrowid = None
    self._rows = None