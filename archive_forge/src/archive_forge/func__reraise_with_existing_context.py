import sys
import warnings
from collections import deque
from functools import wraps
def _reraise_with_existing_context(exc_details):
    exc_type, exc_value, exc_tb = exc_details
    exec('raise exc_type, exc_value, exc_tb')