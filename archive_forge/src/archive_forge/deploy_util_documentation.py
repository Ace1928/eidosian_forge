import logging
import shutil
import sys
from pathlib import Path
from . import EXE_FORMAT
from .config import Config
from .python_helper import PythonExecutable

        Copy the executable into the final location
        For Android deployment, this is done through buildozer
    