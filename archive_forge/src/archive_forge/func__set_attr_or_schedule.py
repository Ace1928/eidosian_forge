import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, cast
from .backends import get_array_ops
from .config import registry
from .types import FloatsXd, Generator
def _set_attr_or_schedule(self, name, value):
    if isinstance(value, (float, bool, int)):
        setattr(self, name, value)
    else:
        if isinstance(value, list):
            value = iter(value)
        self.schedules[name] = value
        try:
            setattr(self, name, next(value))
        except (StopIteration, TypeError) as e:
            err = f"Invalid schedule for '{name}' ({type(value)})\n{e}"
            raise ValueError(err)