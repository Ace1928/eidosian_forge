from __future__ import annotations
import asyncio
import base64
import copy
import json
import mimetypes
import os
import pkgutil
import secrets
import shutil
import tempfile
import warnings
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal, Optional, TypedDict
import fsspec.asyn
import httpx
import huggingface_hub
from huggingface_hub import SpaceStage
from websockets.legacy.protocol import WebSocketCommonProtocol
def apply_diff(obj, diff):
    obj = copy.deepcopy(obj)

    def apply_edit(target, path, action, value):
        if len(path) == 0:
            if action == 'replace':
                return value
            elif action == 'append':
                return target + value
            else:
                raise ValueError(f'Unsupported action: {action}')
        current = target
        for i in range(len(path) - 1):
            current = current[path[i]]
        last_path = path[-1]
        if action == 'replace':
            current[last_path] = value
        elif action == 'append':
            current[last_path] += value
        elif action == 'add':
            if isinstance(current, list):
                current.insert(int(last_path), value)
            else:
                current[last_path] = value
        elif action == 'delete':
            if isinstance(current, list):
                del current[int(last_path)]
            else:
                del current[last_path]
        else:
            raise ValueError(f'Unknown action: {action}')
        return target
    for action, path, value in diff:
        obj = apply_edit(obj, path, action, value)
    return obj