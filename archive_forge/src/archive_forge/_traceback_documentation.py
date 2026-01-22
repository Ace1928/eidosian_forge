from types import TracebackType
from typing import List, Optional
import tempfile
import traceback
import contextlib
import inspect
import os.path

        Bulk version of CapturedTraceback.format.  Returns a list of list of strings.
        