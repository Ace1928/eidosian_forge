from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
@staticmethod
def _remote_exec_chain(num_returns: int, *args: Tuple) -> List[Any]:
    """
        Execute the deconstructed chain in a worker process.

        Parameters
        ----------
        num_returns : int
            The number of return values.
        *args : tuple
            A deconstructed chain to be executed.

        Returns
        -------
        list
            The execution results. The last element of this list is the ``MetaList``.
        """
    if num_returns == 2:
        return _remote_exec_single_chain.remote(*args)
    else:
        return _remote_exec_multi_chain.options(num_returns=num_returns).remote(num_returns, *args)