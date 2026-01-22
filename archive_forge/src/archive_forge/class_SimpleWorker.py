import contextlib
import errno
import logging
import math
import os
import random
import signal
import socket
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from enum import Enum
from random import shuffle
from types import FrameType
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union
from uuid import uuid4
from contextlib import suppress
import redis.exceptions
from . import worker_registration
from .command import PUBSUB_CHANNEL_TEMPLATE, handle_command, parse_payload
from .connections import get_current_connection, pop_connection, push_connection
from .defaults import (
from .exceptions import DequeueTimeout, DeserializationError, ShutDownImminentException
from .job import Job, JobStatus
from .logutils import blue, green, setup_loghandlers, yellow
from .maintenance import clean_intermediate_queue
from .queue import Queue
from .registry import StartedJobRegistry, clean_registries
from .scheduler import RQScheduler
from .serializers import resolve_serializer
from .suspension import is_suspended
from .timeouts import HorseMonitorTimeoutException, JobTimeoutException, UnixSignalDeathPenalty
from .utils import as_text, backend_class, compact, ensure_list, get_version, utcformat, utcnow, utcparse
from .version import VERSION
class SimpleWorker(Worker):

    def execute_job(self, job: 'Job', queue: 'Queue'):
        """Execute job in same thread/process, do not fork()"""
        self.set_state(WorkerStatus.BUSY)
        self.perform_job(job, queue)
        self.set_state(WorkerStatus.IDLE)

    def get_heartbeat_ttl(self, job: 'Job') -> int:
        """-1" means that jobs never timeout. In this case, we should _not_ do -1 + 60 = 59.
        We should just stick to DEFAULT_WORKER_TTL.

        Args:
            job (Job): The Job

        Returns:
            ttl (int): TTL
        """
        if job.timeout == -1:
            return DEFAULT_WORKER_TTL
        else:
            return (job.timeout or DEFAULT_WORKER_TTL) + 60