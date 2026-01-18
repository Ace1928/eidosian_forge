import abc
import contextlib
import functools
import itertools
import threading
from oslo_utils import excutils
from oslo_utils import timeutils
from taskflow.conductors import base
from taskflow import exceptions as excp
from taskflow.listeners import logging as logging_listener
from taskflow import logging
from taskflow import states
from taskflow.types import timing as tt
from taskflow.utils import iter_utils
from taskflow.utils import misc
Waits for the conductor to gracefully exit.

        This method waits for the conductor to gracefully exit. An optional
        timeout can be provided, which will cause the method to return
        within the specified timeout. If the timeout is reached, the returned
        value will be ``False``, otherwise it will be ``True``.

        :param timeout: Maximum number of seconds that the :meth:`wait` method
                        should block for.
        