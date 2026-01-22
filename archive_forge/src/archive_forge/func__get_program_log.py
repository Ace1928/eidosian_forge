import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _get_program_log(program_id: int) -> str:
    """Query a ShaderProgram link logs."""
    result = c_int(0)
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, byref(result))
    result_str = create_string_buffer(result.value)
    glGetProgramInfoLog(program_id, result, None, result_str)
    if result_str.value:
        return f'OpenGL returned the following message when linking the program: \n{result_str.value}'
    else:
        return f"Program '{program_id}' linked successfully."