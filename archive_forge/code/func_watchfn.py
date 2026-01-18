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
def watchfn(reloader: SourceFileReloader):
    """Watch python files in a given module.

    get_changes is taken from uvicorn's default file watcher.
    """
    from gradio.cli.commands.reload import reload_thread
    reload_thread.running_reload = True

    def get_changes() -> Path | None:
        for file in iter_py_files():
            try:
                mtime = file.stat().st_mtime
            except OSError:
                continue
            old_time = mtimes.get(file)
            if old_time is None:
                mtimes[file] = mtime
                continue
            elif mtime > old_time:
                return file
        return None

    def iter_py_files() -> Iterator[Path]:
        for reload_dir in reload_dirs:
            for path in list(reload_dir.rglob('*.py')):
                yield path.resolve()
            for path in list(reload_dir.rglob('*.css')):
                yield path.resolve()
    reload_dirs = [Path(dir_) for dir_ in reloader.watch_dirs]
    import sys
    for dir_ in reload_dirs:
        sys.path.insert(0, str(dir_))
    mtimes = {}
    module = importlib.import_module(reloader.watch_module_name)
    while reloader.should_watch():
        changed = get_changes()
        if changed:
            print(f'Changes detected in: {changed}')
            try:
                changed_in_copy = _remove_no_reload_codeblocks(str(changed))
                if changed != reloader.demo_file:
                    changed_module = _find_module(changed)
                    exec(changed_in_copy, changed_module.__dict__)
                    top_level_parent = sys.modules[changed_module.__name__.split('.')[0]]
                    if top_level_parent != changed_module:
                        importlib.reload(top_level_parent)
                changed_demo_file = _remove_no_reload_codeblocks(str(reloader.demo_file))
                exec(changed_demo_file, module.__dict__)
            except Exception:
                print(f'Reloading {reloader.watch_module_name} failed with the following exception: ')
                traceback.print_exc()
                mtimes = {}
                continue
            demo = getattr(module, reloader.demo_name)
            if reloader.queue_changed(demo):
                print("Reloading failed. The new demo has a queue and the old one doesn't (or vice versa). Please launch your demo again")
            else:
                reloader.swap_blocks(demo)
            mtimes = {}
        time.sleep(0.05)