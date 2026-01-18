from typing import Callable, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
from ..types import ListXd, Ragged
Transform sequences from a ragged format into lists.