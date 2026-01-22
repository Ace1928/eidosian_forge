from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import exceptions as kms_exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import maps
def CheckKeyIsExternalVpc(self, key_version, messages):
    if key_version.protectionLevel != messages.CryptoKeyVersion.ProtectionLevelValueValuesEnum.EXTERNAL_VPC:
        raise kms_exceptions.UpdateError('EkmConnection key path updates are only available for key versions with EXTERNAL_VPC protection level')