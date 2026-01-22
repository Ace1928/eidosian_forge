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
class LegacyIntType(pa.PyExtensionType):

    def __init__(self):
        pa.PyExtensionType.__init__(self, pa.int8())

    def __reduce__(self):
        return (LegacyIntType, ())