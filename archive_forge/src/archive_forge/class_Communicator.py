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
@dataclass
class Communicator:
    """Helper class to help communicate between the worker thread and main thread."""
    lock: Lock
    job: JobStatus
    prediction_processor: Callable[..., tuple]
    reset_url: str
    should_cancel: bool = False
    event_id: str | None = None