from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsVolumeTemplate(_messages.Message):
    """Configuration template for AWS EBS volumes.

  Enums:
    VolumeTypeValueValuesEnum: Optional. Type of the EBS volume. When
      unspecified, it defaults to GP2 volume.

  Fields:
    iops: Optional. The number of I/O operations per second (IOPS) to
      provision for GP3 volume.
    kmsKeyArn: Optional. The Amazon Resource Name (ARN) of the Customer
      Managed Key (CMK) used to encrypt AWS EBS volumes. If not specified, the
      default Amazon managed key associated to the AWS region where this
      cluster runs will be used.
    sizeGib: Optional. The size of the volume, in GiBs. When unspecified, a
      default value is provided. See the specific reference in the parent
      resource.
    throughput: Optional. The throughput that the volume supports, in MiB/s.
      Only valid if volume_type is GP3. If the volume_type is GP3 and this is
      not speficied, it defaults to 125.
    volumeType: Optional. Type of the EBS volume. When unspecified, it
      defaults to GP2 volume.
  """

    class VolumeTypeValueValuesEnum(_messages.Enum):
        """Optional. Type of the EBS volume. When unspecified, it defaults to GP2
    volume.

    Values:
      VOLUME_TYPE_UNSPECIFIED: Not set.
      GP2: GP2 (General Purpose SSD volume type).
      GP3: GP3 (General Purpose SSD volume type).
    """
        VOLUME_TYPE_UNSPECIFIED = 0
        GP2 = 1
        GP3 = 2
    iops = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    kmsKeyArn = _messages.StringField(2)
    sizeGib = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    throughput = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    volumeType = _messages.EnumField('VolumeTypeValueValuesEnum', 5)