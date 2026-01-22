from __future__ import annotations
import asyncio
import logging
import signal
import sys
from typing import Any
from gunicorn.arbiter import Arbiter
from gunicorn.workers.base import Worker
from uvicorn.config import Config
from uvicorn.main import Server
class UvicornH11Worker(UvicornWorker):
    CONFIG_KWARGS = {'loop': 'asyncio', 'http': 'h11'}