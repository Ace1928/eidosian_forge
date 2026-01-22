import inspect
from functools import partial
from typing import (
class ReprError(Exception):
    """An error occurred when attempting to build a repr."""