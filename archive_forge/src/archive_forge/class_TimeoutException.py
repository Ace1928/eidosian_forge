import itertools
from contextlib import ExitStack
class TimeoutException(Exception):
    """Timeout expired"""