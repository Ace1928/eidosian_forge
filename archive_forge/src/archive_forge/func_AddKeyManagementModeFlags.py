from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
def AddKeyManagementModeFlags(parser):
    """Adds key-management-mode flags and related flags."""
    group = parser.add_group(help='Specifies the key management mode for the EkmConnection and associated fields.')
    group.add_argument('--key-management-mode', choices=['manual', 'cloud-kms'], help='Key management mode of the ekm connection. An EkmConnection in `cloud-kms` mode means Cloud KMS will attempt to create and manage the key material that resides on the EKM for crypto keys created with this EkmConnection. An EkmConnection in `manual` mode means the external key material will not be managed by Cloud KMS. Omitting the flag defaults to `manual`.')
    group.add_argument('--crypto-space-path', help='Crypto space path for the EkmConnection. Required during EkmConnection creation if `--key-management-mode=cloud-kms`.')