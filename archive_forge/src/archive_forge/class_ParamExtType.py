import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
class ParamExtType(pa.ExtensionType):

    def __init__(self, width):
        self._width = width
        super().__init__(pa.binary(width), 'pyarrow.tests.test_cffi.ParamExtType')

    @property
    def width(self):
        return self._width

    def __arrow_ext_serialize__(self):
        return str(self.width).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        width = int(serialized.decode())
        return cls(width)