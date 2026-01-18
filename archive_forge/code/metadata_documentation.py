import json
from collections import deque
from numbers import Number
from typing import Tuple, Optional
from ray.train._internal.checkpoint_manager import _CheckpointManager
from ray.tune.utils.serialization import TuneFunctionEncoder, TuneFunctionDecoder
Serializable struct for holding runtime trial metadata.

    Runtime metadata is data that changes and is updated on runtime. This includes
    e.g. the last result, the currently available checkpoints, and the number
    of errors encountered for a trial.
    