import sys
import queue
import threading
from pyglet import app
from pyglet import clock
from pyglet import event
@staticmethod
def _redraw_windows(dt):
    for window in app.windows:
        window.switch_to()
        window.dispatch_event('on_draw')
        window.dispatch_event('on_refresh', dt)
        window.flip()