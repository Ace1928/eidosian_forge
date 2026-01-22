from functools import partial
from . import functions
from . import rpc_async
import torch
from .constants import UNSET_RPC_TIMEOUT
from torch.futures import Future
def _complete_op(fut):
    try:
        result.set_result(fut.value())
    except BaseException as ex:
        result.set_exception(ex)