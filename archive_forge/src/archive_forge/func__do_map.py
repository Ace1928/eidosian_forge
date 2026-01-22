import traceback
import eventlet
from eventlet import queue
from eventlet.support import greenlets as greenlet
def _do_map(self, func, it, gi):
    for args in it:
        gi.spawn(func, *args)
    gi.done_spawning()