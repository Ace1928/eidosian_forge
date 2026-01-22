from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
@staticmethod
def _GkeNodeConfigFromArgPool(dataproc, arg_pool):
    """Creates the GkeNodeConfig via the arguments specified in --pools."""
    pool = dataproc.messages.GkeNodeConfig()
    if 'machineType' in arg_pool:
        pool.machineType = arg_pool['machineType']
    if 'preemptible' in arg_pool:
        pool.preemptible = arg_pool['preemptible']
    if 'localSsdCount' in arg_pool:
        pool.localSsdCount = arg_pool['localSsdCount']
    if 'localNvmeSsdCount' in arg_pool:
        pool.ephemeralStorageConfig = dataproc.messages.GkeEphemeralStorageConfig(localSsdCount=arg_pool['localNvmeSsdCount'])
    if 'accelerators' in arg_pool:
        pool.accelerators = _GkeNodePoolTargetParser._GkeNodePoolAcceleratorConfigFromArgPool(dataproc, arg_pool['accelerators'])
    if 'minCpuPlatform' in arg_pool:
        pool.minCpuPlatform = arg_pool['minCpuPlatform']
    if 'bootDiskKmsKey' in arg_pool:
        pool.bootDiskKmsKey = arg_pool['bootDiskKmsKey']
    if pool != dataproc.messages.GkeNodeConfig():
        return pool
    return None