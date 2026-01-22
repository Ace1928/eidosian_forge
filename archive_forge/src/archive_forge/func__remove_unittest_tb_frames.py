import io
import sys
import traceback
from . import util
from functools import wraps
def _remove_unittest_tb_frames(self, tb):
    """Truncates usercode tb at the first unittest frame.

        If the first frame of the traceback is in user code,
        the prefix up to the first unittest frame is returned.
        If the first frame is already in the unittest module,
        the traceback is not modified.
        """
    prev = None
    while tb and (not self._is_relevant_tb_level(tb)):
        prev = tb
        tb = tb.tb_next
    if prev is not None:
        prev.tb_next = None