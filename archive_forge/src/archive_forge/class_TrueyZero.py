import collections
import threading
import time
import socket
import warnings
import queue
from jaraco.functools import pass_none
class TrueyZero:
    """Object which equals and does math like the integer 0 but evals True."""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other