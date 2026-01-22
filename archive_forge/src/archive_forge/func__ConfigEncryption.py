from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _ConfigEncryption(self, args):
    kwargs = {'kmsKeyArn': aws_flags.GetConfigEncryptionKmsKeyArn(args)}
    return self._messages.GoogleCloudGkemulticloudV1AwsConfigEncryption(**kwargs) if any(kwargs.values()) else None