from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def _error_received(self, error_args):
    self.expecting = 'end'
    self.request_handler.post_body_error_received(error_args)