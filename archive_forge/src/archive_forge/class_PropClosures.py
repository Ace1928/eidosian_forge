from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class PropClosures:

    def __init__(self):
        self.bases = {}
        self.lock = None

    def set_threaded(self):
        if self.lock is None:
            import threading
            self.lock = threading.Lock()

    def get(self, ctx):
        if self.lock:
            with self.lock:
                r = self.bases[ctx]
        else:
            r = self.bases[ctx]
        return r

    def set(self, ctx, r):
        if self.lock:
            with self.lock:
                self.bases[ctx] = r
        else:
            self.bases[ctx] = r

    def insert(self, r):
        if self.lock:
            with self.lock:
                id = len(self.bases) + 3
                self.bases[id] = r
        else:
            id = len(self.bases) + 3
            self.bases[id] = r
        return id