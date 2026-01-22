from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.arrays.formathandler import FormatHandler
from OpenGL.raw.GL import _types 
from OpenGL import error
from OpenGL._bytes import bytes,unicode,as_8_bit
import ctypes,logging
from OpenGL._bytes import long, integer_types
import weakref
from OpenGL import acceleratesupport
class VBO(object):
    """Instances can be passed into array-handling routines

        You can check for whether VBOs are supported by accessing the implementation:

            if bool(vbo.get_implementation()):
                # vbo version of code
            else:
                # fallback version of code
        """
    copied = False
    _no_cache_ = True

    def __init__(self, data, usage='GL_DYNAMIC_DRAW', target='GL_ARRAY_BUFFER', size=None):
        """Initialize the VBO object 
            
            data -- PyOpenGL-compatible array-data structure, numpy arrays, ctypes arrays, etc.
            usage -- OpenGL usage constant describing expected data-flow patterns (this is a hint 
                to the GL about where/how to cache the data)
                
                GL_STATIC_DRAW_ARB
                GL_STATIC_READ_ARB
                GL_STATIC_COPY_ARB
                GL_DYNAMIC_DRAW_ARB
                GL_DYNAMIC_READ_ARB
                GL_DYNAMIC_COPY_ARB
                GL_STREAM_DRAW_ARB
                GL_STREAM_READ_ARB
                GL_STREAM_COPY_ARB
                
                DRAW constants suggest to the card that the data will be primarily used to draw 
                on the card.  READ that the data will be read back into the GL.  COPY means that 
                the data will be used both for DRAW and READ operations.
                
                STATIC suggests that the data will only be written once (or a small number of times).
                DYNAMIC suggests that the data will be used a small number of times before being 
                discarded.
                STREAM suggests that the data will be updated approximately every time that it is 
                used (that is, it will likely only be used once).
                
            target -- VBO target to which to bind (array or indices)
                GL_ARRAY_BUFFER -- array-data binding 
                GL_ELEMENT_ARRAY_BUFFER -- index-data binding
                GL_UNIFORM_BUFFER -- used to pass mid-size arrays of data packed into a buffer
                GL_TEXTURE_BUFFER -- used to pass large arrays of data as a pseudo-texture
                GL_TRANSFORM_FEEDBACK_BUFFER -- used to receive transformed vertices for processing
                
            size -- if not provided, will use arrayByteCount to determine the size of the data-array,
                thus this value (number of bytes) is required when using opaque data-structures,
                (such as ctypes pointers) as the array data-source.
            """
        self.usage = usage
        self.set_array(data, size)
        self.target = target
        self.buffers = []
        self._copy_segments = []
    _I_ = None
    implementation = property(get_implementation)

    def resolve(self, value):
        """Resolve string constant to constant"""
        if isinstance(value, (bytes, unicode)):
            return getattr(self.implementation, self.implementation.basename(value))
        return value

    def set_array(self, data, size=None):
        """Update our entire array with new data
            
            data -- PyOpenGL-compatible array-data structure, numpy arrays, ctypes arrays, etc.
            size -- if not provided, will use arrayByteCount to determine the size of the data-array,
                thus this value (number of bytes) is required when using opaque data-structures,
                (such as ctypes pointers) as the array data-source.
            """
        self.data = data
        self.copied = False
        if size is not None:
            self.size = size
        elif self.data is not None:
            self.size = ArrayDatatype.arrayByteCount(self.data)

    def __setitem__(self, slice, array):
        """Set slice of data on the array and vbo (if copied already)

            slice -- the Python slice object determining how the data should
                be copied into the vbo/array
            array -- something array-compatible that will be used as the
                source of the data, note that the data-format will have to
                be the same as the internal data-array to work properly, if
                not, the amount of data copied will be wrong.

            This is a reasonably complex operation, it has to have all sorts
            of state-aware changes to correctly map the source into the low-level
            OpenGL view of the buffer (which is just bytes as far as the GL
            is concerned).
            """
        if slice.step and (not slice.step == 1):
            raise NotImplemented("Don't know how to map stepped arrays yet")
        data = ArrayDatatype.asArray(array)
        data_length = ArrayDatatype.arrayByteCount(array)
        start = slice.start or 0
        stop = slice.stop or len(self.data)
        if start < 0:
            start += len(self.data)
            start = max((start, 0))
        if stop < 0:
            stop += len(self.data)
            stop = max((stop, 0))
        self.data[slice] = data
        if self.copied and self.buffers:
            if start - stop == len(self.data):
                self.copied = False
            elif len(data):
                size = ArrayDatatype.arrayByteCount(self.data[0])
                start *= size
                stop *= size
                self._copy_segments.append((start, stop - start, data))

    def __len__(self):
        """Delegate length/truth checks to our data-array"""
        return len(self.data)

    def __getattr__(self, key):
        """Delegate failing attribute lookups to our data-array"""
        if key not in ('data', 'usage', 'target', 'buffers', 'copied', '_I_', 'implementation', '_copy_segments'):
            return getattr(self.data, key)
        else:
            raise AttributeError(key)

    def create_buffers(self):
        """Create the internal buffer(s)"""
        assert not self.buffers, 'Already created the buffer'
        self.buffers = [long(self.implementation.glGenBuffers(1))]
        self.target = self.resolve(self.target)
        self.usage = self.resolve(self.usage)
        self.implementation._DELETERS_[id(self)] = weakref.ref(self, self.implementation.deleter(self.buffers, id(self)))
        return self.buffers

    def copy_data(self):
        """Copy our data into the buffer on the GL side (if required)
            
            Ensures that the GL's version of the data in the VBO matches our 
            internal view of the data, either by copying the entire data-set 
            over with glBufferData or by updating the already-transferred 
            data with glBufferSubData.
            """
        assert self.buffers, 'Should do create_buffers before copy_data'
        if self.copied:
            if self._copy_segments:
                while self._copy_segments:
                    start, size, data = self._copy_segments.pop(0)
                    dataptr = ArrayDatatype.voidDataPointer(data)
                    self.implementation.glBufferSubData(self.target, start, size, dataptr)
        else:
            if self.data is not None and self.size is None:
                self.size = ArrayDatatype.arrayByteCount(self.data)
            self.implementation.glBufferData(self.target, self.size, self.data, self.usage)
            self.copied = True

    def delete(self):
        """Delete this buffer explicitly"""
        if self.buffers:
            while self.buffers:
                try:
                    self.implementation.glDeleteBuffers(1, self.buffers.pop(0))
                except (AttributeError, error.NullFunctionError) as err:
                    pass

    def __int__(self):
        """Get our VBO id"""
        if not self.buffers:
            self.create_buffers()
        return self.buffers[0]

    def bind(self):
        """Bind this buffer for use in vertex calls
            
            If we have not yet created our implementation-level VBO, then we 
            will create it before binding.  Once bound, calls self.copy_data()
            """
        if not self.buffers:
            buffers = self.create_buffers()
        self.implementation.glBindBuffer(self.target, self.buffers[0])
        self.copy_data()

    def unbind(self):
        """Unbind the buffer (make normal array operations active)"""
        self.implementation.glBindBuffer(self.target, 0)

    def __add__(self, other):
        """Add an integer to this VBO (create a VBOOffset)"""
        if hasattr(other, 'offset'):
            other = other.offset
        assert isinstance(other, integer_types), 'Only know how to add integer/long offsets'
        return VBOOffset(self, other)
    __enter__ = bind

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Context manager exit"""
        self.unbind()
        return False