import collections
import datetime
import logging
import queue as queue_module
import threading
import time
from google.api_core import exceptions
def _never_terminate(future_or_error):
    """By default, no errors cause BiDi termination."""
    return False