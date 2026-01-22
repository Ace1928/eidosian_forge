import contextlib
import errno
import logging
import os
import signal
import time
from enum import Enum
from multiprocessing import Process
from typing import Dict, List, NamedTuple, Optional, Type, Union
from uuid import uuid4
from redis import ConnectionPool, Redis
from rq.serializers import DefaultSerializer
from .connections import parse_connection
from .defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT
from .job import Job
from .logutils import setup_loghandlers
from .queue import Queue
from .utils import parse_names
from .worker import BaseWorker, Worker
def all_workers_have_stopped(self) -> bool:
    """Returns True if all workers have stopped."""
    self.reap_workers()
    return self.number_of_active_workers == 0