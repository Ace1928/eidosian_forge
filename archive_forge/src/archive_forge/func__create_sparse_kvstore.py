import os
import time
import logging
import warnings
from collections import namedtuple
import numpy as np
from . import io
from . import ndarray as nd
from . import symbol as sym
from . import optimizer as opt
from . import metric
from . import kvstore as kvs
from .context import Context, cpu
from .initializer import Uniform
from .optimizer import get_updater
from .executor_manager import DataParallelExecutorManager, _check_arguments, _load_data
from .io import DataDesc
from .base import mx_real_t
from .callback import LogValidationMetricsCallback # pylint: disable=wrong-import-position
def _create_sparse_kvstore(kvstore):
    """Create kvstore assuming some parameters' storage types are row_sparse.

    Parameters
    ----------
    kvstore : KVStore or str
        The kvstore.

    Returns
    -------
    kvstore : KVStore
    update_on_kvstore : bool. Always True.
    """
    if isinstance(kvstore, kvs.KVStore):
        kv = kvstore
    elif isinstance(kvstore, str):
        kv = kvs.create(kvstore)
    else:
        raise TypeError("Cannot create '%s' KVStore with row_sparse parameters. The type must be KVStore or str." % kvstore)
    assert kv.is_capable(kvs.KVStoreBase.OPTIMIZER), 'KVStore with sparse weight requires optimizer support. However, type(kv) does not support optimizer. Please consider other kvstore backends (e.g. dist_device) instead.'
    return (kv, True)