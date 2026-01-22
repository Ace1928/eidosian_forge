import array
import contextlib
import enum
import struct
@property
def AsInt(self):
    """Returns current reference as integer value."""
    if self.IsNull:
        return 0
    elif self.IsBool:
        return int(self.AsBool)
    elif self._type is Type.INT:
        return _Unpack(I, self._Bytes)
    elif self._type is Type.INDIRECT_INT:
        return _Unpack(I, self._Indirect()[:self._byte_width])
    if self._type is Type.UINT:
        return _Unpack(U, self._Bytes)
    elif self._type is Type.INDIRECT_UINT:
        return _Unpack(U, self._Indirect()[:self._byte_width])
    elif self.IsString:
        return len(self.AsString)
    elif self.IsKey:
        return len(self.AsKey)
    elif self.IsBlob:
        return len(self.AsBlob)
    elif self.IsVector:
        return len(self.AsVector)
    elif self.IsTypedVector:
        return len(self.AsTypedVector)
    elif self.IsFixedTypedVector:
        return len(self.AsFixedTypedVector)
    else:
        raise self._ConvertError(Type.INT)