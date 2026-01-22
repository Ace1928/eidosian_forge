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
def append_unique_suffix(name: str, list_of_names: list[str]):
    """Appends a numerical suffix to `name` so that it does not appear in `list_of_names`."""
    set_of_names: set[str] = set(list_of_names)
    if name not in set_of_names:
        return name
    else:
        suffix_counter = 1
        new_name = f'{name}_{suffix_counter}'
        while new_name in set_of_names:
            suffix_counter += 1
            new_name = f'{name}_{suffix_counter}'
        return new_name