from typing import Callable, Optional, Tuple, TypeVar, Union, cast
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array3d, ArrayXd, ListXd, Padded, Ragged
Transform sequence data into a contiguous array on the way into and
    out of a model. Handles a variety of sequence types: lists, padded and ragged.
    If the input is an array, it is passed through unchanged.
    