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
def validate_queue_settings(self):
    for dep in self.dependencies:
        for i in dep['cancels']:
            if not self.queue_enabled_for_fn(i):
                raise ValueError('Queue needs to be enabled! You may get this error by either 1) passing a function that uses the yield keyword into an interface without enabling the queue or 2) defining an event that cancels another event without enabling the queue. Both can be solved by calling .queue() before .launch()')
        if dep['batch'] and dep['queue'] is False:
            raise ValueError('In order to use batching, the queue must be enabled.')