import inspect
import functools
from enum import Enum
import torch.autograd
def _set_datapipe_valid_iterator_id(datapipe):
    """Given a DataPipe, updates its valid iterator ID and reset the DataPipe."""
    if hasattr(datapipe, '_is_child_datapipe') and datapipe._is_child_datapipe is True:
        if hasattr(datapipe, '_set_main_datapipe_valid_iterator_id'):
            datapipe._set_main_datapipe_valid_iterator_id()
        else:
            raise RuntimeError('ChildDataPipe must have method `_set_main_datapipe_valid_iterator_id`.')
    else:
        if datapipe._valid_iterator_id is None:
            datapipe._valid_iterator_id = 0
        else:
            datapipe._valid_iterator_id += 1
        datapipe.reset()
    return datapipe._valid_iterator_id