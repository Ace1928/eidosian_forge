from __future__ import annotations
import logging
import os
import signal
import threading
from multiprocessing.context import SpawnProcess
from socket import socket
from types import FrameType
from typing import Callable
import click
from uvicorn._subprocess import get_subprocess
from uvicorn.config import Config

        A signal handler that is registered with the parent process.
        