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
def test_null_storage_type():
    ext_type = AnnotatedType(pa.null(), {'key': 'value'})
    storage = pa.array([None] * 10, pa.null())
    arr = pa.ExtensionArray.from_storage(ext_type, storage)
    assert arr.null_count == 10
    arr.validate(full=True)