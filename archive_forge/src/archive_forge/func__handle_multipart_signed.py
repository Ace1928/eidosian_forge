import re
import sys
import time
import random
from copy import deepcopy
from io import StringIO, BytesIO
from email.utils import _has_surrogates
def _handle_multipart_signed(self, msg):
    p = self.policy
    self.policy = p.clone(max_line_length=0)
    try:
        self._handle_multipart(msg)
    finally:
        self.policy = p