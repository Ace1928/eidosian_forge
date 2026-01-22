import re
from io import BytesIO
from .. import errors
def _reset_current_record(self):
    self._current_record_length = None
    self._current_record_names = []