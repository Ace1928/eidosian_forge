from __future__ import annotations
import os.path
import threading
from typing import Any
from streamlit import config
from streamlit.runtime.scriptrunner import magic
from streamlit.source_util import open_python_file
Return the bytecode for the Python script at the given path.

        If the bytecode is not already in the cache, the script will be
        compiled first.

        Raises
        ------
        Any Exception raised while reading or compiling the script.

        Notes
        -----
        Threading: SAFE. May be called on any thread.
        