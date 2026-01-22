import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
class IntegerEmbeddedType(pa.ExtensionType):

    def __init__(self):
        super().__init__(IntegerType(), 'pyarrow.tests.IntegerType')

    def __arrow_ext_serialize__(self):
        return self.storage_type.__arrow_ext_serialize__()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        deserialized_storage_type = storage_type.__arrow_ext_deserialize__(serialized)
        assert deserialized_storage_type == storage_type
        return cls()