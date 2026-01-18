import inspect
import re
import typing as T
from collections import OrderedDict, namedtuple
from enum import IntEnum
from .common import (
Parse the Google-style docstring into its components.

        :returns: parsed docstring
        