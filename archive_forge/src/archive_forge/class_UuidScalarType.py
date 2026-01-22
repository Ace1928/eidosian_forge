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
class UuidScalarType(pa.ExtensionScalar):

    def as_py(self):
        return None if self.value is None else UUID(bytes=self.value.as_py())