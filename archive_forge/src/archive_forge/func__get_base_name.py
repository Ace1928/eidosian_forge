from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import numpy.typing as npt
def _get_base_name(param: str) -> str:
    return param.split('.')[0].split(':')[0]