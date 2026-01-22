import pickle
import os
import zlib
import inspect
from io import BytesIO
from .numpy_pickle_utils import _ZFILE_PREFIX
from .numpy_pickle_utils import Unpickler
from .numpy_pickle_utils import _ensure_native_byte_order
class NDArrayWrapper(object):
    """An object to be persisted instead of numpy arrays.

    The only thing this object does, is to carry the filename in which
    the array has been persisted, and the array subclass.
    """

    def __init__(self, filename, subclass, allow_mmap=True):
        """Constructor. Store the useful information for later."""
        self.filename = filename
        self.subclass = subclass
        self.allow_mmap = allow_mmap

    def read(self, unpickler):
        """Reconstruct the array."""
        filename = os.path.join(unpickler._dirname, self.filename)
        allow_mmap = getattr(self, 'allow_mmap', True)
        kwargs = {}
        if allow_mmap:
            kwargs['mmap_mode'] = unpickler.mmap_mode
        if 'allow_pickle' in inspect.signature(unpickler.np.load).parameters:
            kwargs['allow_pickle'] = True
        array = unpickler.np.load(filename, **kwargs)
        array = _ensure_native_byte_order(array)
        if hasattr(array, '__array_prepare__') and self.subclass not in (unpickler.np.ndarray, unpickler.np.memmap):
            new_array = unpickler.np.core.multiarray._reconstruct(self.subclass, (0,), 'b')
            return new_array.__array_prepare__(array)
        else:
            return array