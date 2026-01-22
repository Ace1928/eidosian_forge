import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def _append_replay(self, replay: Replay) -> List[Replay]:
    replays = copy.deepcopy(self.replays)
    replays.append(replay)
    return replays