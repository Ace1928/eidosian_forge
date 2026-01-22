import time
import logging
import mxnet as mx
from mxnet.module import Module
from .svrg_optimizer import _SVRGOptimizer
def _allocate_gradients(self, key, value):
    """Allocate average of full gradients accumulated in the KVStore to each device.

        Parameters
        ----------

        key: int or str
            Key in the kvstore.
        value: List of NDArray, List of RowSparseNDArray
            A list of average of the full gradients in the KVStore.
        """
    for i in range(self._ctx_len):
        self._param_dict[i][key] = value[i] / self._ctx_len