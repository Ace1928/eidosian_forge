from __future__ import annotations
import asyncio
import concurrent.futures
import inspect
import itertools
import logging
import os
import socket
import sys
import threading
import time
import typing as t
import uuid
import warnings
from datetime import datetime
from functools import partial
from signal import SIGINT, SIGTERM, Signals, default_int_handler, signal
from .control import CONTROL_THREAD_NAME
import psutil
import zmq
from IPython.core.error import StdinNotImplementedError
from jupyter_client.session import Session
from tornado import ioloop
from tornado.queues import Queue, QueueEmpty
from traitlets.config.configurable import SingletonConfigurable
from traitlets.traitlets import (
from zmq.eventloop.zmqstream import ZMQStream
from ipykernel.jsonutil import json_clean
from ._version import kernel_protocol_version
from .iostream import OutStream
def _accepts_parameters(meth, param_names):
    parameters = inspect.signature(meth).parameters
    accepts = {param: False for param in param_names}
    for param in param_names:
        param_spec = parameters.get(param)
        accepts[param] = param_spec and param_spec.kind in [param_spec.KEYWORD_ONLY, param_spec.POSITIONAL_OR_KEYWORD] or any((p.kind == p.VAR_KEYWORD for p in parameters.values()))
    return accepts