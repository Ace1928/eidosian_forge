import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
class ComputeShaderProgram:
    """OpenGL Compute Shader Program"""

    def __init__(self, source: str):
        """Create an OpenGL ComputeShaderProgram from source."""
        self._id = None
        if not (gl_info.have_version(4, 3) or gl_info.have_extension('GL_ARB_compute_shader')):
            raise ShaderException("Compute Shader not supported. OpenGL Context version must be at least 4.3 or higher, or 4.2 with the 'GL_ARB_compute_shader' extension.")
        self._shader = Shader(source, 'compute')
        self._context = pyglet.gl.current_context
        self._id = _link_program(self._shader)
        if _debug_gl_shaders:
            print(_get_program_log(self._id))
        self._uniforms = _introspect_uniforms(self._id, True)
        self._uniform_blocks = _introspect_uniform_blocks(self)
        self.max_work_group_size = self._get_tuple(GL_MAX_COMPUTE_WORK_GROUP_SIZE)
        self.max_work_group_count = self._get_tuple(GL_MAX_COMPUTE_WORK_GROUP_COUNT)
        self.max_shared_memory_size = self._get_value(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE)
        self.max_work_group_invocations = self._get_value(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)

    @staticmethod
    def _get_tuple(parameter: int):
        val_x = GLint()
        val_y = GLint()
        val_z = GLint()
        for i, value in enumerate((val_x, val_y, val_z)):
            glGetIntegeri_v(parameter, i, byref(value))
        return (val_x.value, val_y.value, val_z.value)

    @staticmethod
    def _get_value(parameter: int) -> int:
        val = GLint()
        glGetIntegerv(parameter, byref(val))
        return val.value

    @staticmethod
    def dispatch(x: int=1, y: int=1, z: int=1, barrier: int=GL_ALL_BARRIER_BITS) -> None:
        """Launch one or more compute work groups.

        The ComputeShaderProgram should be active (bound) before calling
        this method. The x, y, and z parameters specify the number of local
        work groups that will be  dispatched in the X, Y and Z dimensions.
        """
        glDispatchCompute(x, y, z)
        if barrier:
            glMemoryBarrier(barrier)

    @property
    def id(self) -> int:
        return self._id

    @property
    def uniforms(self):
        return {n: dict(location=u.location, length=u.length, size=u.size) for n, u in self._uniforms.items()}

    @property
    def uniform_blocks(self) -> dict:
        return self._uniform_blocks

    def use(self) -> None:
        glUseProgram(self._id)

    @staticmethod
    def stop():
        glUseProgram(0)
    __enter__ = use
    bind = use
    unbind = stop

    def __exit__(self, *_):
        glUseProgram(0)

    def delete(self):
        glDeleteProgram(self._id)
        self._id = None

    def __del__(self):
        if self._id is not None:
            try:
                self._context.delete_shader_program(self._id)
                self._id = None
            except (AttributeError, ImportError):
                pass

    def __setitem__(self, key, value):
        try:
            uniform = self._uniforms[key]
        except KeyError as err:
            msg = f'A Uniform with the name `{key}` was not found.\nThe spelling may be incorrect, or if not in use it may have been optimized out by the OpenGL driver.'
            if _debug_gl_shaders:
                warnings.warn(msg)
                return
            else:
                raise ShaderException() from err
        try:
            uniform.set(value)
        except GLException as err:
            raise ShaderException from err

    def __getitem__(self, item):
        try:
            uniform = self._uniforms[item]
        except KeyError as err:
            msg = f'A Uniform with the name `{item}` was not found.\nThe spelling may be incorrect, or if not in use it may have been optimized out by the OpenGL driver.'
            if _debug_gl_shaders:
                warnings.warn(msg)
                return
            else:
                raise ShaderException(msg) from err
        try:
            return uniform.get()
        except GLException as err:
            raise ShaderException from err