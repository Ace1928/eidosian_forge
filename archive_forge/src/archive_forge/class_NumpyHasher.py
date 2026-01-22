import pickle
import hashlib
import sys
import types
import struct
import io
import decimal
class NumpyHasher(Hasher):
    """ Special case the hasher for when numpy is loaded.
    """

    def __init__(self, hash_name='md5', coerce_mmap=False):
        """
            Parameters
            ----------
            hash_name: string
                The hash algorithm to be used
            coerce_mmap: boolean
                Make no difference between np.memmap and np.ndarray
                objects.
        """
        self.coerce_mmap = coerce_mmap
        Hasher.__init__(self, hash_name=hash_name)
        import numpy as np
        self.np = np
        if hasattr(np, 'getbuffer'):
            self._getbuffer = np.getbuffer
        else:
            self._getbuffer = memoryview

    def save(self, obj):
        """ Subclass the save method, to hash ndarray subclass, rather
            than pickling them. Off course, this is a total abuse of
            the Pickler class.
        """
        if isinstance(obj, self.np.ndarray) and (not obj.dtype.hasobject):
            if obj.shape == ():
                obj_c_contiguous = obj.flatten()
            elif obj.flags.c_contiguous:
                obj_c_contiguous = obj
            elif obj.flags.f_contiguous:
                obj_c_contiguous = obj.T
            else:
                obj_c_contiguous = obj.flatten()
            self._hash.update(self._getbuffer(obj_c_contiguous.view(self.np.uint8)))
            if self.coerce_mmap and isinstance(obj, self.np.memmap):
                klass = self.np.ndarray
            else:
                klass = obj.__class__
            obj = (klass, ('HASHED', obj.dtype, obj.shape, obj.strides))
        elif isinstance(obj, self.np.dtype):
            self._hash.update('_HASHED_DTYPE'.encode('utf-8'))
            self._hash.update(pickle.dumps(obj))
            return
        Hasher.save(self, obj)