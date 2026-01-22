import abc
import atexit
import contextlib
import logging
import os
import pathlib
import random
import tempfile
import time
import typing
import warnings
from . import constants, exceptions, portalocker
class NamedBoundedSemaphore(BoundedSemaphore):
    """
    Bounded semaphore to prevent too many parallel processes from running

    It's also possible to specify a timeout when acquiring the lock to wait
    for a resource to become available.  This is very similar to
    `threading.BoundedSemaphore` but works across multiple processes and across
    multiple operating systems.

    Because this works across multiple processes it's important to give the
    semaphore a name.  This name is used to create the lock files.  If you
    don't specify a name, a random name will be generated.  This means that
    you can't use the same semaphore in multiple processes unless you pass the
    semaphore object to the other processes.

    >>> semaphore = NamedBoundedSemaphore(2, name='test')
    >>> str(semaphore.get_filenames()[0])
    '...test.00.lock'

    >>> semaphore = NamedBoundedSemaphore(2)
    >>> 'bounded_semaphore' in str(semaphore.get_filenames()[0])
    True

    """

    def __init__(self, maximum: int, name: typing.Optional[str]=None, filename_pattern: str='{name}.{number:02d}.lock', directory: str=tempfile.gettempdir(), timeout: typing.Optional[float]=DEFAULT_TIMEOUT, check_interval: typing.Optional[float]=DEFAULT_CHECK_INTERVAL, fail_when_locked: typing.Optional[bool]=True):
        if name is None:
            name = 'bounded_semaphore.%d' % random.randint(0, 1000000)
        super().__init__(maximum, name, filename_pattern, directory, timeout, check_interval, fail_when_locked)