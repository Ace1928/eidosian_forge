import os
import sys
import threading
from . import process
from . import reduction
def Barrier(self, parties, action=None, timeout=None):
    """Returns a barrier object"""
    from .synchronize import Barrier
    return Barrier(parties, action, timeout, ctx=self.get_context())