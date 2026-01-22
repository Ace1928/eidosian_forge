import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
def _check_caught_exception(self, attempt_number, caught_exception):
    raise caught_exception