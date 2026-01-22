from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _ServicesAuthentication(self, args):
    kwargs = {'roleArn': aws_flags.GetRoleArn(args), 'roleSessionName': aws_flags.GetRoleSessionName(args)}
    if not any(kwargs.values()):
        return None
    return self._messages.GoogleCloudGkemulticloudV1AwsServicesAuthentication(**kwargs)