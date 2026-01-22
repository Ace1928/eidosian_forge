from collections import defaultdict
from functools import partial
from threading import Lock
import inspect
import warnings
import logging
from transitions.core import Machine, Event, listify
class IdentManager:
    """  Manages the identity of threads to detect whether the current thread already has a lock. """

    def __init__(self):
        self.current = 0

    def __enter__(self):
        self.current = get_ident()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.current = 0