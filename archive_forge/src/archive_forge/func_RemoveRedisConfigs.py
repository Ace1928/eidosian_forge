from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
def RemoveRedisConfigs(instance_ref, args, patch_request):
    if not getattr(patch_request.instance, 'redisConfigs', None):
        return patch_request
    if args.IsSpecified('remove_redis_config'):
        config_dict = encoding.MessageToDict(patch_request.instance.redisConfigs)
        for removed_key in args.remove_redis_config:
            config_dict.pop(removed_key, None)
        patch_request = AddNewRedisConfigs(instance_ref, config_dict, patch_request)
    return patch_request