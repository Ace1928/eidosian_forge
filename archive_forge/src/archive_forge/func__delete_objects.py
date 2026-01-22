import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def _delete_objects(self, list_, deletion_func):
    """Release all OpenGL objects in the given list using the supplied
        deletion function with the signature ``(GLuint count, GLuint *names)``.
        """
    count = len(list_)
    to_delete = list_[:count]
    del list_[:count]
    deletion_func(count, (gl.GLuint * count)(*to_delete))