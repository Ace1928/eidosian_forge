from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _VolumeTemplate(self, args, kind):
    kwargs = {}
    if kind == 'main':
        kwargs['iops'] = aws_flags.GetMainVolumeIops(args)
        kwargs['kmsKeyArn'] = aws_flags.GetMainVolumeKmsKeyArn(args)
        kwargs['sizeGib'] = flags.GetMainVolumeSize(args)
        kwargs['volumeType'] = aws_flags.GetMainVolumeType(args)
        kwargs['throughput'] = aws_flags.GetMainVolumeThroughput(args)
    elif kind == 'root':
        kwargs['iops'] = aws_flags.GetRootVolumeIops(args)
        kwargs['kmsKeyArn'] = aws_flags.GetRootVolumeKmsKeyArn(args)
        kwargs['sizeGib'] = flags.GetRootVolumeSize(args)
        kwargs['volumeType'] = aws_flags.GetRootVolumeType(args)
        kwargs['throughput'] = aws_flags.GetRootVolumeThroughput(args)
    return self._messages.GoogleCloudGkemulticloudV1AwsVolumeTemplate(**kwargs) if any(kwargs.values()) else None