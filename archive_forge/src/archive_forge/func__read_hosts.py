import logging
import os
import socket
from typing import Dict, List
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.utilities.cloud_io import get_filesystem
@staticmethod
def _read_hosts() -> List[str]:
    """Read compute hosts that are a part of the compute job.

        LSF uses the Job Step Manager (JSM) to manage job steps. Job steps are executed by the JSM from "launch" nodes.
        Each job is assigned a launch node. This launch node will be the first node in the list contained in
        ``LSB_DJOB_RANKFILE``.

        """
    var = 'LSB_DJOB_RANKFILE'
    rankfile = os.environ.get(var)
    if rankfile is None:
        raise ValueError('Did not find the environment variable `LSB_DJOB_RANKFILE`')
    if not rankfile:
        raise ValueError('The environment variable `LSB_DJOB_RANKFILE` is empty')
    fs = get_filesystem(rankfile)
    with fs.open(rankfile, 'r') as f:
        ret = [line.strip() for line in f]
    return ret[1:]