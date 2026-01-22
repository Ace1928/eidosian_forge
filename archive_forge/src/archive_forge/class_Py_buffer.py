import ctypes,sys
from ._arrayconstants import *
class Py_buffer(ctypes.Structure):
    """Wrapper around the Python buffer structure..."""

    @classmethod
    def from_object(cls, object, flags=PyBUF_STRIDES | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS):
        """Create a new Py_buffer referencing ram of object"""
        if not CheckBuffer(object):
            raise TypeError('%s type does not support Buffer Protocol' % (object.__class__,))
        buf = cls()
        result = GetBuffer(object, buf, flags)
        if result != 0:
            raise ValueError('Unable to retrieve Buffer from %s' % (object,))
        if not buf.buf:
            raise ValueError('Null pointer result from %s' % (object,))
        return buf
    _fields_ = _fields_

    @property
    def dims(self):
        return self.shape[:self.ndim]

    def __len__(self):
        return self.shape[0]

    @property
    def dim_strides(self):
        if self.strides:
            return self.strides[:self.ndim]
        return None

    def __enter__(self):
        pass

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        if self.obj:
            ReleaseBuffer(self)

    def __del__(self):
        if self.obj:
            ReleaseBuffer(self)