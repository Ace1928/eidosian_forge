import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _introspect_uniform_blocks(program) -> dict:
    uniform_blocks = {}
    program_id = program.id
    for index in range(_get_number(program_id, GL_ACTIVE_UNIFORM_BLOCKS)):
        name = _get_uniform_block_name(program_id, index)
        num_active = GLint()
        block_data_size = GLint()
        glGetActiveUniformBlockiv(program_id, index, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, num_active)
        glGetActiveUniformBlockiv(program_id, index, GL_UNIFORM_BLOCK_DATA_SIZE, block_data_size)
        indices = (GLuint * num_active.value)()
        indices_ptr = cast(addressof(indices), POINTER(GLint))
        glGetActiveUniformBlockiv(program_id, index, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, indices_ptr)
        uniforms = {}
        for block_uniform_index in indices:
            uniform_name, u_type, u_size = _query_uniform(program_id, block_uniform_index)
            try:
                _, uniform_name = uniform_name.split('.')
            except ValueError:
                pass
            gl_type, _, _, length = _uniform_setters[u_type]
            uniforms[block_uniform_index] = (uniform_name, gl_type, length)
        uniform_blocks[name] = UniformBlock(program, name, index, block_data_size.value, uniforms)
        glUniformBlockBinding(program_id, index, index)
        if _debug_gl_shaders:
            for block in uniform_blocks.values():
                print(f' Found uniform block: {block}')
    return uniform_blocks