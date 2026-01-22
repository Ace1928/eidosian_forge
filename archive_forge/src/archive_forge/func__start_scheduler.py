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
def _start_scheduler(self, burst: bool=False, logging_level: str='INFO', date_format: str=DEFAULT_LOGGING_DATE_FORMAT, log_format: str=DEFAULT_LOGGING_FORMAT):
    """Starts the scheduler process.
        This is specifically designed to be run by the worker when running the `work()` method.
        Instanciates the RQScheduler and tries to acquire a lock.
        If the lock is acquired, start scheduler.
        If worker is on burst mode just enqueues scheduled jobs and quits,
        otherwise, starts the scheduler in a separate process.

        Args:
            burst (bool, optional): Whether to work on burst mode. Defaults to False.
            logging_level (str, optional): Logging level to use. Defaults to "INFO".
            date_format (str, optional): Date Format. Defaults to DEFAULT_LOGGING_DATE_FORMAT.
            log_format (str, optional): Log Format. Defaults to DEFAULT_LOGGING_FORMAT.
        """
    self.scheduler = RQScheduler(self.queues, connection=self.connection, logging_level=logging_level, date_format=date_format, log_format=log_format, serializer=self.serializer)
    self.scheduler.acquire_locks()
    if self.scheduler.acquired_locks:
        if burst:
            self.scheduler.enqueue_scheduled_jobs()
            self.scheduler.release_locks()
        else:
            self.scheduler.start()