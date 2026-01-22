import importlib
import logging
import os
import types
from pathlib import Path
import torch
from torchaudio._internal.module_utils import eval_env
def _init_dll_path():
    for path in os.environ.get('PATH', '').split(';'):
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass