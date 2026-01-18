import inspect
import warnings
from typing import Any, List, Optional, Set
import torch
from torch.utils.data.datapipes.iter.sharding import (
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps

    Traverse the graph of ``DataPipes`` to find random ``DataPipe`` with an API of ``set_seed``.

    Then set the random seed based on the provided RNG to those ``DataPipe``.

    Args:
        datapipe: DataPipe that needs to set randomness
        rng: Random number generator to generate random seeds
    