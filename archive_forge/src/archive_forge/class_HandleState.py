from PySide6.QtCore import (QCoreApplication, QDateTime, QDeadlineTimer,
from . import futures
from . import tasks
import asyncio
import collections.abc
import concurrent.futures
import contextvars
import enum
import os
import signal
import socket
import subprocess
import typing
import warnings
class HandleState(enum.Enum):
    PENDING = enum.auto()
    CANCELLED = enum.auto()
    DONE = enum.auto()