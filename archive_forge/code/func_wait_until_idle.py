import socket
import sys
from collections import defaultdict
from functools import partial
from itertools import count
from typing import Any, Callable, Dict, Sequence, TextIO, Tuple  # noqa
from kombu.exceptions import ContentDisallowed
from kombu.utils.functional import retry_over_time
from celery import states
from celery.exceptions import TimeoutError
from celery.result import AsyncResult, ResultSet  # noqa
from celery.utils.text import truncate
from celery.utils.time import humanize_seconds as _humanize_seconds
def wait_until_idle(self):
    control = self.app.control
    with self.app.connection() as connection:
        while True:
            count = control.purge(connection=connection)
            if count == 0:
                break
        inspect = control.inspect()
        inspect.connection = connection
        while True:
            try:
                count = sum((len(t) for t in inspect.active().values()))
            except ContentDisallowed:
                break
            if count == 0:
                break