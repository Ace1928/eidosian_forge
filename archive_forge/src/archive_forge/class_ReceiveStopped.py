from concurrent.futures import Future
from contextlib import contextmanager
import functools
import os
from selectors import EVENT_READ
import socket
from queue import Queue, Full as QueueFull
from threading import Lock, Thread
from typing import Optional
from jeepney import Message, MessageType
from jeepney.bus import get_bus
from jeepney.bus_messages import message_bus
from jeepney.wrappers import ProxyBase, unwrap_msg
from .blocking import (
from .common import (
class ReceiveStopped(Exception):
    pass