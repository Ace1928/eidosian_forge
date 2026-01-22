from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def _wait_for_response_end(self):
    while not self.finished_reading:
        self._read_more()