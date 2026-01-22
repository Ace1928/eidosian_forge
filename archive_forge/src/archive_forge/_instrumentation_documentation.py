import logging
import types
from typing import Any, Callable, Dict, Sequence, TypeVar
from .._abc import Instrument
Call hookname(*args) on each applicable instrument.

        You must first check whether there are any instruments installed for
        that hook, e.g.::

            if "before_task_step" in instruments:
                instruments.call("before_task_step", task)
        