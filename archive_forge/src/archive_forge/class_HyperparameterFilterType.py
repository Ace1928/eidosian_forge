from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class HyperparameterFilterType(enum.Enum):
    """Describes how to represent filter values."""
    REGEX = 'regex'
    INTERVAL = 'interval'
    DISCRETE = 'discrete'