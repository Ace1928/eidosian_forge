import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _query_uniform(program_id: int, index: int):
    """Query the name, type, and size of a Uniform by index."""
    usize = GLint()
    utype = GLenum()
    buf_size = 192
    uname = create_string_buffer(buf_size)
    try:
        glGetActiveUniform(program_id, index, buf_size, None, usize, utype, uname)
        return (uname.value.decode(), utype.value, usize.value)
    except GLException as exc:
        raise ShaderException from exc