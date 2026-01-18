import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def vertex_list_indexed(self, count, mode, indices, batch=None, group=None, **data):
    """Create a IndexedVertexList.

        :Parameters:
            `count` : int
                The number of vertices in the list.
            `mode` : int
                OpenGL drawing mode enumeration; for example, one of
                ``GL_POINTS``, ``GL_LINES``, ``GL_TRIANGLES``, etc.
                This determines how the list is drawn in the given batch.
            `indices` : sequence of int
                Sequence of integers giving indices into the vertex list.
            `batch` : `~pyglet.graphics.Batch`
                Batch to add the VertexList to, or ``None`` if a Batch will not be used.
                Using a Batch is strongly recommended.
            `group` : `~pyglet.graphics.Group`
                Group to add the VertexList to, or ``None`` if no group is required.
            `**data` : str or tuple
                Attribute formats and initial data for the vertex list.

        :rtype: :py:class:`~pyglet.graphics.vertexdomain.IndexedVertexList`
        """
    attributes = self._attributes.copy()
    initial_arrays = []
    for name, fmt in data.items():
        try:
            if isinstance(fmt, tuple):
                fmt, array = fmt
                initial_arrays.append((name, array))
            attributes[name] = {**attributes[name], **{'format': fmt}}
        except KeyError:
            raise ShaderException(f'An attribute with the name `{name}` was not found. Please check the spelling.\nIf the attribute is not in use in the program, it may have been optimized out by the OpenGL driver.\nValid names: \n{list(attributes)}')
    batch = batch or pyglet.graphics.get_default_batch()
    domain = batch.get_domain(True, mode, group, self, attributes)
    vlist = domain.create(count, len(indices))
    start = vlist.start
    vlist.indices = [i + start for i in indices]
    for name, array in initial_arrays:
        vlist.set_attribute_data(name, array)
    return vlist