import pickle
import os
import warnings
import io
from pathlib import Path
from .compressor import lz4, LZ4_NOT_INSTALLED_ERROR
from .compressor import _COMPRESSORS, register_compressor, BinaryZlibFile
from .compressor import (ZlibCompressorWrapper, GzipCompressorWrapper,
from .numpy_pickle_utils import Unpickler, Pickler
from .numpy_pickle_utils import _read_fileobject, _write_fileobject
from .numpy_pickle_utils import _read_bytes, BUFFER_SIZE
from .numpy_pickle_utils import _ensure_native_byte_order
from .numpy_pickle_compat import load_compatibility
from .numpy_pickle_compat import NDArrayWrapper
from .numpy_pickle_compat import ZNDArrayWrapper  # noqa
from .backports import make_memmap
class NumpyArrayWrapper(object):
    """An object to be persisted instead of numpy arrays.

    This object is used to hack into the pickle machinery and read numpy
    array data from our custom persistence format.
    More precisely, this object is used for:
    * carrying the information of the persisted array: subclass, shape, order,
    dtype. Those ndarray metadata are used to correctly reconstruct the array
    with low level numpy functions.
    * determining if memmap is allowed on the array.
    * reading the array bytes from a file.
    * reading the array using memorymap from a file.
    * writing the array bytes to a file.

    Attributes
    ----------
    subclass: numpy.ndarray subclass
        Determine the subclass of the wrapped array.
    shape: numpy.ndarray shape
        Determine the shape of the wrapped array.
    order: {'C', 'F'}
        Determine the order of wrapped array data. 'C' is for C order, 'F' is
        for fortran order.
    dtype: numpy.ndarray dtype
        Determine the data type of the wrapped array.
    allow_mmap: bool
        Determine if memory mapping is allowed on the wrapped array.
        Default: False.
    """

    def __init__(self, subclass, shape, order, dtype, allow_mmap=False, numpy_array_alignment_bytes=NUMPY_ARRAY_ALIGNMENT_BYTES):
        """Constructor. Store the useful information for later."""
        self.subclass = subclass
        self.shape = shape
        self.order = order
        self.dtype = dtype
        self.allow_mmap = allow_mmap
        self.numpy_array_alignment_bytes = numpy_array_alignment_bytes

    def safe_get_numpy_array_alignment_bytes(self):
        return getattr(self, 'numpy_array_alignment_bytes', None)

    def write_array(self, array, pickler):
        """Write array bytes to pickler file handle.

        This function is an adaptation of the numpy write_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        buffersize = max(16 * 1024 ** 2 // array.itemsize, 1)
        if array.dtype.hasobject:
            pickle.dump(array, pickler.file_handle, protocol=2)
        else:
            numpy_array_alignment_bytes = self.safe_get_numpy_array_alignment_bytes()
            if numpy_array_alignment_bytes is not None:
                current_pos = pickler.file_handle.tell()
                pos_after_padding_byte = current_pos + 1
                padding_length = numpy_array_alignment_bytes - pos_after_padding_byte % numpy_array_alignment_bytes
                padding_length_byte = int.to_bytes(padding_length, length=1, byteorder='little')
                pickler.file_handle.write(padding_length_byte)
                if padding_length != 0:
                    padding = b'\xff' * padding_length
                    pickler.file_handle.write(padding)
            for chunk in pickler.np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'], buffersize=buffersize, order=self.order):
                pickler.file_handle.write(chunk.tobytes('C'))

    def read_array(self, unpickler):
        """Read array from unpickler file handle.

        This function is an adaptation of the numpy read_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        if len(self.shape) == 0:
            count = 1
        else:
            shape_int64 = [unpickler.np.int64(x) for x in self.shape]
            count = unpickler.np.multiply.reduce(shape_int64)
        if self.dtype.hasobject:
            array = pickle.load(unpickler.file_handle)
        else:
            numpy_array_alignment_bytes = self.safe_get_numpy_array_alignment_bytes()
            if numpy_array_alignment_bytes is not None:
                padding_byte = unpickler.file_handle.read(1)
                padding_length = int.from_bytes(padding_byte, byteorder='little')
                if padding_length != 0:
                    unpickler.file_handle.read(padding_length)
            max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, self.dtype.itemsize)
            array = unpickler.np.empty(count, dtype=self.dtype)
            for i in range(0, count, max_read_count):
                read_count = min(max_read_count, count - i)
                read_size = int(read_count * self.dtype.itemsize)
                data = _read_bytes(unpickler.file_handle, read_size, 'array data')
                array[i:i + read_count] = unpickler.np.frombuffer(data, dtype=self.dtype, count=read_count)
                del data
            if self.order == 'F':
                array.shape = self.shape[::-1]
                array = array.transpose()
            else:
                array.shape = self.shape
        return _ensure_native_byte_order(array)

    def read_mmap(self, unpickler):
        """Read an array using numpy memmap."""
        current_pos = unpickler.file_handle.tell()
        offset = current_pos
        numpy_array_alignment_bytes = self.safe_get_numpy_array_alignment_bytes()
        if numpy_array_alignment_bytes is not None:
            padding_byte = unpickler.file_handle.read(1)
            padding_length = int.from_bytes(padding_byte, byteorder='little')
            offset += padding_length + 1
        if unpickler.mmap_mode == 'w+':
            unpickler.mmap_mode = 'r+'
        marray = make_memmap(unpickler.filename, dtype=self.dtype, shape=self.shape, order=self.order, mode=unpickler.mmap_mode, offset=offset)
        unpickler.file_handle.seek(offset + marray.nbytes)
        if numpy_array_alignment_bytes is None and current_pos % NUMPY_ARRAY_ALIGNMENT_BYTES != 0:
            message = f'The memmapped array {marray} loaded from the file {unpickler.file_handle.name} is not byte aligned. This may cause segmentation faults if this memmapped array is used in some libraries like BLAS or PyTorch. To get rid of this warning, regenerate your pickle file with joblib >= 1.2.0. See https://github.com/joblib/joblib/issues/563 for more details'
            warnings.warn(message)
        return _ensure_native_byte_order(marray)

    def read(self, unpickler):
        """Read the array corresponding to this wrapper.

        Use the unpickler to get all information to correctly read the array.

        Parameters
        ----------
        unpickler: NumpyUnpickler

        Returns
        -------
        array: numpy.ndarray

        """
        if unpickler.mmap_mode is not None and self.allow_mmap:
            array = self.read_mmap(unpickler)
        else:
            array = self.read_array(unpickler)
        if hasattr(array, '__array_prepare__') and self.subclass not in (unpickler.np.ndarray, unpickler.np.memmap):
            new_array = unpickler.np.core.multiarray._reconstruct(self.subclass, (0,), 'b')
            return new_array.__array_prepare__(array)
        else:
            return array