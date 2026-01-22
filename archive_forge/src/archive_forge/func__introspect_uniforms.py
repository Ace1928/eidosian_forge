import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _introspect_uniforms(self):
    """Introspect the block's structure and return a ctypes struct for
        manipulating the uniform block's members.
        """
    p_id = self.program.id
    index = self.index
    active_count = len(self.uniforms)
    indices = (GLuint * active_count)()
    offsets = (GLint * active_count)()
    indices_ptr = cast(addressof(indices), POINTER(GLint))
    offsets_ptr = cast(addressof(offsets), POINTER(GLint))
    glGetActiveUniformBlockiv(p_id, index, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, indices_ptr)
    glGetActiveUniformsiv(p_id, active_count, indices, GL_UNIFORM_OFFSET, offsets_ptr)
    _oi = sorted(zip(offsets, indices), key=lambda x: x[0])
    offsets = [x[0] for x in _oi] + [self.size]
    indices = (GLuint * active_count)(*(x[1] for x in _oi))
    view_fields = []
    for i in range(active_count):
        u_name, gl_type, length = self.uniforms[indices[i]]
        size = offsets[i + 1] - offsets[i]
        c_type_size = sizeof(gl_type)
        actual_size = c_type_size * length
        padding = size - actual_size
        arg = (u_name, gl_type * length) if length > 1 else (u_name, gl_type)
        view_fields.append(arg)
        if padding > 0:
            padding_bytes = padding // c_type_size
            view_fields.append((f'_padding{i}', gl_type * padding_bytes))

    class View(Structure):
        _fields_ = view_fields

        def __repr__(self):
            return str(dict(self._fields_))
    return View