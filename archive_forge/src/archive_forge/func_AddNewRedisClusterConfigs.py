from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import cluster_util
from googlecloudsdk.command_lib.redis import util
def AddNewRedisClusterConfigs(cluster_ref, redis_configs_dict, patch_request):
    messages = util.GetMessagesForResource(cluster_ref)
    new_redis_configs = cluster_util.PackageClusterRedisConfig(redis_configs_dict, messages)
    patch_request.cluster.redisConfigs = new_redis_configs
    patch_request = AddFieldToUpdateMask('redis_configs', patch_request)
    return patch_request