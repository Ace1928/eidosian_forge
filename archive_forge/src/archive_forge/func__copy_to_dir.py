from __future__ import annotations
import pathlib
import secrets
import shutil
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from fastapi import Request
from gradio_client.utils import traverse
from . import wasm_utils
def _copy_to_dir(self, dir: str) -> FileData:
    pathlib.Path(dir).mkdir(exist_ok=True)
    new_obj = dict(self)
    if not self.path:
        raise ValueError('Source file path is not set')
    new_name = shutil.copy(self.path, dir)
    new_obj['path'] = new_name
    return self.__class__(**new_obj)