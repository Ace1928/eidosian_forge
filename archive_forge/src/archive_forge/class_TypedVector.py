import array
import contextlib
import enum
import struct
class TypedVector(Sized):
    """Data accessor for the encoded typed vector or fixed typed vector bytes."""
    __slots__ = ('_element_type', '_size')

    def __init__(self, buf, byte_width, element_type, size=0):
        super().__init__(buf, byte_width, size)
        if element_type == Type.STRING:
            element_type = Type.KEY
        self._element_type = element_type

    @property
    def Bytes(self):
        return self._buf[:self._byte_width * len(self)]

    @property
    def ElementType(self):
        return self._element_type

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError('vector index %s is out of [0, %d) range' % (index, len(self)))
        buf = self._buf.Slice(index * self._byte_width)
        return Ref(buf, self._byte_width, 1, self._element_type)

    @property
    def Value(self):
        """Returns underlying data as list object."""
        if not self:
            return []
        if self._element_type is Type.BOOL:
            return [bool(e) for e in _UnpackVector(U, self.Bytes, len(self))]
        elif self._element_type is Type.INT:
            return list(_UnpackVector(I, self.Bytes, len(self)))
        elif self._element_type is Type.UINT:
            return list(_UnpackVector(U, self.Bytes, len(self)))
        elif self._element_type is Type.FLOAT:
            return list(_UnpackVector(F, self.Bytes, len(self)))
        elif self._element_type is Type.KEY:
            return [e.AsKey for e in self]
        elif self._element_type is Type.STRING:
            return [e.AsString for e in self]
        else:
            raise TypeError('unsupported element_type: %s' % self._element_type)

    def __repr__(self):
        return 'TypedVector(%s, byte_width=%d, element_type=%s, size=%d)' % (self._buf, self._byte_width, self._element_type, self._size)