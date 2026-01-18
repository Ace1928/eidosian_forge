import io
import pickle
import warnings
from collections.abc import Collection
from typing import Dict, List, Optional, Set, Tuple, Type, Union
from torch.utils.data import IterDataPipe, MapDataPipe
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def traverse_dps(datapipe: DataPipe) -> DataPipeGraph:
    """
    Traverse the DataPipes and their attributes to extract the DataPipe graph.

    This only looks into the attribute from each DataPipe that is either a
    DataPipe and a Python collection object such as ``list``, ``tuple``,
    ``set`` and ``dict``.

    Args:
        datapipe: the end DataPipe of the graph
    Returns:
        A graph represented as a nested dictionary, where keys are ids of DataPipe instances
        and values are tuples of DataPipe instance and the sub-graph
    """
    cache: Set[int] = set()
    return _traverse_helper(datapipe, only_datapipe=True, cache=cache)