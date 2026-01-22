import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _link_program(*shaders) -> int:
    """Link one or more Shaders into a ShaderProgram."""
    program_id = glCreateProgram()
    for shader in shaders:
        glAttachShader(program_id, shader.id)
    glLinkProgram(program_id)
    status = c_int()
    glGetProgramiv(program_id, GL_LINK_STATUS, byref(status))
    if not status.value:
        length = c_int()
        glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, length)
        log = c_buffer(length.value)
        glGetProgramInfoLog(program_id, len(log), None, log)
        raise ShaderException('Error linking shader program:\n{}'.format(log.value.decode()))
    for shader in shaders:
        glDetachShader(program_id, shader.id)
    return program_id