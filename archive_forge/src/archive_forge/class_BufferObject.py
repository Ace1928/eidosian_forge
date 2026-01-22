import sys
import ctypes
from functools import lru_cache
import pyglet
from pyglet.gl import *
class BufferObject(AbstractBuffer):
    """Lightweight representation of an OpenGL Buffer Object.

    The data in the buffer is not replicated in any system memory (unless it
    is done so by the video driver).  While this can improve memory usage and
    possibly performance, updates to the buffer are relatively slow.
    The target of the buffer is ``GL_ARRAY_BUFFER`` internally to avoid
    accidentally overriding other states when altering the buffer contents.
    The intended target can be set when binding the buffer.

    This class does not implement :py:class:`AbstractMappable`, and so has no
    :py:meth:`~AbstractMappable.get_region` method.  See 
    :py:class:`MappableVertexBufferObject` for a Buffer class
    that does implement :py:meth:`~AbstractMappable.get_region`.
    """

    def __init__(self, size, usage=GL_DYNAMIC_DRAW):
        self.size = size
        self.usage = usage
        self._context = pyglet.gl.current_context
        buffer_id = GLuint()
        glGenBuffers(1, buffer_id)
        self.id = buffer_id.value
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        data = (GLubyte * self.size)()
        glBufferData(GL_ARRAY_BUFFER, self.size, data, self.usage)

    def invalidate(self):
        glBufferData(GL_ARRAY_BUFFER, self.size, None, self.usage)

    def bind(self, target=GL_ARRAY_BUFFER):
        glBindBuffer(target, self.id)

    def unbind(self):
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def bind_to_index_buffer(self):
        """Binds this buffer as an index buffer on the active vertex array."""
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.id)

    def set_data(self, data):
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glBufferData(GL_ARRAY_BUFFER, self.size, data, self.usage)

    def set_data_region(self, data, start, length):
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glBufferSubData(GL_ARRAY_BUFFER, start, length, data)

    def map(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        ptr = ctypes.cast(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY), ctypes.POINTER(ctypes.c_byte * self.size)).contents
        return ptr

    def map_range(self, start, size, ptr_type):
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        ptr = ctypes.cast(glMapBufferRange(GL_ARRAY_BUFFER, start, size, GL_MAP_WRITE_BIT), ptr_type).contents
        return ptr

    def unmap(self):
        glUnmapBuffer(GL_ARRAY_BUFFER)

    def delete(self):
        glDeleteBuffers(1, GLuint(self.id))
        self.id = None

    def __del__(self):
        if self.id is not None:
            try:
                self._context.delete_buffer(self.id)
                self.id = None
            except (AttributeError, ImportError):
                pass

    def resize(self, size):
        temp = (ctypes.c_byte * size)()
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        data = glMapBufferRange(GL_ARRAY_BUFFER, 0, self.size, GL_MAP_READ_BIT)
        ctypes.memmove(temp, data, min(size, self.size))
        glUnmapBuffer(GL_ARRAY_BUFFER)
        self.size = size
        glBufferData(GL_ARRAY_BUFFER, self.size, temp, self.usage)

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id}, size={self.size})'