import os
import sys
import threading
from . import process
from . import reduction
def Event(self):
    """Returns an event object"""
    from .synchronize import Event
    return Event(ctx=self.get_context())