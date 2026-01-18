import collections
import queue
import torch
from . import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper
Contains definitions of the methods used by the _BaseDataLoaderIter to put fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
