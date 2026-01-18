from __future__ import annotations
import copy
import hashlib
import inspect
import json
import os
import random
import secrets
import string
import sys
import threading
import time
import warnings
import webbrowser
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Literal, Sequence, cast
from urllib.parse import urlparse, urlunparse
import anyio
import fastapi
import httpx
from anyio import CapacityLimiter
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import (
from gradio.blocks_events import BlocksEvents, BlocksMeta
from gradio.context import Context
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.events import (
from gradio.exceptions import (
from gradio.helpers import create_tracker, skip, special_args
from gradio.state_holder import SessionState
from gradio.themes import Default as DefaultTheme
from gradio.themes import ThemeClass as Theme
from gradio.tunneling import (
from gradio.utils import (
def validate_outputs(self, fn_index: int, predictions: Any | list[Any]):
    block_fn = self.fns[fn_index]
    dependency = self.dependencies[fn_index]
    dep_outputs = dependency['outputs']
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]
    if len(predictions) < len(dep_outputs):
        name = f' ({block_fn.name})' if block_fn.name and block_fn.name != '<lambda>' else ''
        wanted_args = []
        received_args = []
        for output_id in dep_outputs:
            block = self.blocks[output_id]
            wanted_args.append(str(block))
        for pred in predictions:
            v = f'"{pred}"' if isinstance(pred, str) else str(pred)
            received_args.append(v)
        wanted = ', '.join(wanted_args)
        received = ', '.join(received_args)
        raise ValueError(f"An event handler{name} didn't receive enough output values (needed: {len(dep_outputs)}, received: {len(predictions)}).\nWanted outputs:\n    [{wanted}]\nReceived outputs:\n    [{received}]")