import pandas
import ray
from ray.util import get_node_ip_address
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.core.execution.ray.common import RayWrapper
from modin.utils import _inherit_docstrings
from .partition import PandasOnRayDataframePartition
@ray.remote
def _deploy_ray_func(deployer, *positional_args, axis, f_to_deploy, f_len_args, f_kwargs, extract_metadata=True, **kwargs):
    """
    Execute a function on an axis partition in a worker process.

    This is ALWAYS called on either ``PandasDataframeAxisPartition.deploy_axis_func``
    or ``PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions``, which both
    serve to deploy another dataframe function on a Ray worker process. The provided `positional_args`
    contains positional arguments for both: `deployer` and for `f_to_deploy`, the parameters can be separated
    using the `f_len_args` value. The parameters are combined so they will be deserialized by Ray before the
    kernel is executed (`f_kwargs` will never contain more Ray objects, and thus does not require deserialization).

    Parameters
    ----------
    deployer : callable
        A `PandasDataFrameAxisPartition.deploy_*` method that will call ``f_to_deploy``.
    *positional_args : list
        The first `f_len_args` elements in this list represent positional arguments
        to pass to the `f_to_deploy`. The rest are positional arguments that will be
        passed to `deployer`.
    axis : {0, 1}
        The axis to perform the function along. This argument is keyword only.
    f_to_deploy : callable or RayObjectID
        The function to deploy. This argument is keyword only.
    f_len_args : int
        Number of positional arguments to pass to ``f_to_deploy``. This argument is keyword only.
    f_kwargs : dict
        Keyword arguments to pass to ``f_to_deploy``. This argument is keyword only.
    extract_metadata : bool, default: True
        Whether to return metadata (length, width, ip) of the result. Passing `False` may relax
        the load on object storage as the remote function would return 4 times fewer futures.
        Passing `False` makes sense for temporary results where you know for sure that the
        metadata will never be requested. This argument is keyword only.
    **kwargs : dict
        Keyword arguments to pass to ``deployer``.

    Returns
    -------
    list : Union[tuple, list]
        The result of the function call, and metadata for it.

    Notes
    -----
    Ray functions are not detected by codecov (thus pragma: no cover).
    """
    f_args = positional_args[:f_len_args]
    deploy_args = positional_args[f_len_args:]
    result = deployer(axis, f_to_deploy, f_args, f_kwargs, *deploy_args, **kwargs)
    if not extract_metadata:
        for item in result:
            yield item
    else:
        ip = get_node_ip_address()
        for r in result:
            if isinstance(r, pandas.DataFrame):
                for item in [r, len(r), len(r.columns), ip]:
                    yield item
            else:
                for item in [r, None, None, ip]:
                    yield item