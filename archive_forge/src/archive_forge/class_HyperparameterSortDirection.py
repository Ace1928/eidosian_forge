from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class HyperparameterSortDirection(enum.Enum):
    """Describes which direction to sort a value."""
    ASCENDING = 'ascending'
    DESCENDING = 'descending'