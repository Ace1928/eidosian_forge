import logging
import warnings
from enum import IntEnum, unique
from typing import Any, Callable, List, Optional, TypeVar
import numpy as np
from . import collective
A context controlling rabit initialization and finalization.