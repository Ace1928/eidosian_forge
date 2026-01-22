import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def _safe_to_operate_on_object_space(self):
    """Return whether it is safe to interact with this context's object
        space.

        This is considered to be the case if the currently active context's
        object space is the same as this context's object space and this
        method is called from the main thread.
        """
    return self.object_space is gl.current_context.object_space and threading.current_thread() is threading.main_thread()