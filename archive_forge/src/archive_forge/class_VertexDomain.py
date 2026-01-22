import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
class VertexDomain:
    """Management of a set of vertex lists.

    Construction of a vertex domain is usually done with the
    :py:func:`create_domain` function.
    """
    _initial_count = 16

    def __init__(self, program, attribute_meta):
        self.program = program
        self.attribute_meta = attribute_meta
        self.allocator = allocation.Allocator(self._initial_count)
        self.attribute_names = {}
        self.buffer_attributes = []
        self._property_dict = {}
        for name, meta in attribute_meta.items():
            assert meta['format'][0] in _gl_types, f"'{meta['format']}' is not a valid atrribute format for '{name}'."
            location = meta['location']
            count = meta['count']
            gl_type = _gl_types[meta['format'][0]]
            normalize = 'n' in meta['format']
            attribute = shader.Attribute(name, location, count, gl_type, normalize)
            self.attribute_names[attribute.name] = attribute
            attribute.buffer = AttributeBufferObject(attribute.stride * self.allocator.capacity, attribute)
            self.buffer_attributes.append((attribute.buffer, (attribute,)))
            self._property_dict[attribute.name] = _make_attribute_property(name)
        self._vertexlist_class = type('VertexList', (VertexList,), self._property_dict)
        self.vao = vertexarray.VertexArray()
        self.vao.bind()
        for buffer, attributes in self.buffer_attributes:
            buffer.bind()
            for attribute in attributes:
                attribute.enable()
                attribute.set_pointer(buffer.ptr)
        self.vao.unbind()

    def safe_alloc(self, count):
        """Allocate vertices, resizing the buffers if necessary."""
        try:
            return self.allocator.alloc(count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            for buffer, _ in self.buffer_attributes:
                buffer.resize(capacity * buffer.attribute_stride)
            self.allocator.set_capacity(capacity)
            return self.allocator.alloc(count)

    def safe_realloc(self, start, count, new_count):
        """Reallocate vertices, resizing the buffers if necessary."""
        try:
            return self.allocator.realloc(start, count, new_count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            for buffer, _ in self.buffer_attributes:
                buffer.resize(capacity * buffer.attribute_stride)
            self.allocator.set_capacity(capacity)
            return self.allocator.realloc(start, count, new_count)

    def create(self, count, index_count=None):
        """Create a :py:class:`VertexList` in this domain.

        :Parameters:
            `count` : int
                Number of vertices to create.
            `index_count`: None
                Ignored for non indexed VertexDomains

        :rtype: :py:class:`VertexList`
        """
        start = self.safe_alloc(count)
        return self._vertexlist_class(self, start, count)

    def draw(self, mode):
        """Draw all vertices in the domain.

        All vertices in the domain are drawn at once. This is the
        most efficient way to render primitives.

        :Parameters:
            `mode` : int
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.

        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.sub_data()
        starts, sizes = self.allocator.get_allocated_regions()
        primcount = len(starts)
        if primcount == 0:
            pass
        elif primcount == 1:
            glDrawArrays(mode, starts[0], sizes[0])
        else:
            starts = (GLint * primcount)(*starts)
            sizes = (GLsizei * primcount)(*sizes)
            glMultiDrawArrays(mode, starts, sizes, primcount)

    def draw_subset(self, mode, vertex_list):
        """Draw a specific VertexList in the domain.

        The `vertex_list` parameter specifies a :py:class:`VertexList`
        to draw. Only primitives in that list will be drawn.

        :Parameters:
            `mode` : int
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.
            `vertex_list` : `VertexList`
                Vertex list to draw.

        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.sub_data()
        glDrawArrays(mode, vertex_list.start, vertex_list.count)

    @property
    def is_empty(self):
        return not self.allocator.starts

    def __repr__(self):
        return '<%s@%x %s>' % (self.__class__.__name__, id(self), self.allocator)