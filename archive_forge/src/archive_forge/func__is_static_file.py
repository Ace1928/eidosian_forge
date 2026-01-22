from __future__ import annotations
import ast
import asyncio
import copy
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import json
import json.decoder
import os
import pkgutil
import re
import sys
import tempfile
import threading
import time
import traceback
import typing
import urllib.parse
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from io import BytesIO
from numbers import Number
from pathlib import Path
from types import AsyncGeneratorType, GeneratorType, ModuleType
from typing import (
import anyio
import gradio_client.utils as client_utils
import httpx
from gradio_client.documentation import document
from typing_extensions import ParamSpec
import gradio
from gradio.context import Context
from gradio.data_classes import FileData
from gradio.strings import en
def _is_static_file(file_path: Any, static_files: list[Path]) -> bool:
    """
    Returns True if the file is a static file (i.e. is is in the static files list).
    """
    if not isinstance(file_path, (str, Path, FileData)):
        return False
    if isinstance(file_path, FileData):
        file_path = file_path.path
    if isinstance(file_path, str):
        file_path = Path(file_path)
        if not file_path.exists():
            return False
    return any((is_in_or_equal(file_path, static_file) for static_file in static_files))