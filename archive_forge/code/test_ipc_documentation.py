from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
ARROW-15783: Verify to_pandas works for interval types.

    Interval types require static structures to be enabled. This test verifies
    that they are when no other library functions are invoked.
    