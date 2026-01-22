from __future__ import annotations
import sys
import eventlet
def _green_thread_modules():
    from eventlet.green import Queue
    from eventlet.green import thread
    from eventlet.green import threading
    return [('queue', Queue), ('_thread', thread), ('threading', threading)]