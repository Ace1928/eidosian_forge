import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
Preload dependent shared libraries.

    The preload configuration file (cupy/.data/_wheel.json) will be added
    during the wheel build process.
    