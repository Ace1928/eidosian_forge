from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _UpdateSettings(self, args):
    kwargs = {'surgeSettings': self._SurgeSettings(args)}
    return self._messages.GoogleCloudGkemulticloudV1UpdateSettings(**kwargs) if any(kwargs.values()) else None