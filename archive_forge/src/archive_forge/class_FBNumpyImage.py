import warnings
from itertools import product
import numpy as np
import pytest
from ..filebasedimages import FileBasedHeader, FileBasedImage, SerializableImage
from .test_image_api import GenericImageAPI, SerializeMixin
class FBNumpyImage(FileBasedImage):
    header_class = FileBasedHeader
    valid_exts = ('.npy',)
    files_types = (('image', '.npy'),)

    def __init__(self, arr, header=None, extra=None, file_map=None):
        super().__init__(header, extra, file_map)
        self.arr = arr

    @property
    def shape(self):
        return self.arr.shape

    def get_data(self):
        warnings.warn('Deprecated', DeprecationWarning)
        return self.arr

    @property
    def dataobj(self):
        return self.arr

    def get_fdata(self):
        return self.arr.astype(np.float64)

    @classmethod
    def from_file_map(klass, file_map):
        with file_map['image'].get_prepare_fileobj('rb') as fobj:
            arr = np.load(fobj)
        return klass(arr)

    def to_file_map(self, file_map=None):
        file_map = self.file_map if file_map is None else file_map
        with file_map['image'].get_prepare_fileobj('wb') as fobj:
            np.save(fobj, self.arr)

    def get_data_dtype(self):
        return self.arr.dtype

    def set_data_dtype(self, dtype):
        self.arr = self.arr.astype(dtype)