import filecmp
import importlib
import os
import re
import shutil
import signal
import sys
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import try_to_load_from_cache
from .utils import (
def _raise_timeout_error(signum, frame):
    raise ValueError('Loading this model requires you to execute custom code contained in the model repository on your local machine. Please set the option `trust_remote_code=True` to permit loading of this model.')