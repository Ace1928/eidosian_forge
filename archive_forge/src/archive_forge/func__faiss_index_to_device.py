import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
@staticmethod
def _faiss_index_to_device(index: 'faiss.Index', device: Optional[Union[int, List[int]]]=None) -> 'faiss.Index':
    """
        Sends a faiss index to a device.
        A device can either be a positive integer (GPU id), a negative integer (all GPUs),
            or a list of positive integers (select GPUs to use), or `None` for CPU.
        """
    if device is None:
        return index
    import faiss
    if isinstance(device, int):
        if device > -1:
            faiss_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(faiss_res, device, index)
        else:
            index = faiss.index_cpu_to_all_gpus(index)
    elif isinstance(device, (list, tuple)):
        index = faiss.index_cpu_to_gpus_list(index, gpus=list(device))
    else:
        raise TypeError(f'The argument type: {type(device)} is not expected. ' + 'Please pass in either nothing, a positive int, a negative int, or a list of positive ints.')
    return index