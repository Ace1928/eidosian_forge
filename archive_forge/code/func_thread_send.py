import asyncio
import atexit
import time
from concurrent.futures import Future
from functools import partial
from threading import Thread
from typing import Any, Dict, List, Optional
import zmq
from tornado.ioloop import IOLoop
from traitlets import Instance, Type
from traitlets.log import get_logger
from zmq.eventloop import zmqstream
from .channels import HBChannel
from .client import KernelClient
from .session import Session
def thread_send() -> None:
    assert self.session is not None
    self.session.send(self.stream, msg)