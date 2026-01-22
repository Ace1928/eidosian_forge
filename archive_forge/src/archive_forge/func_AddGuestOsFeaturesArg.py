from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def AddGuestOsFeaturesArg(parser, messages, supported_features=None):
    """Add the guest-os-features arg."""
    features_enum_type = messages.GuestOsFeature.TypeValueValuesEnum
    excluded_enums = ['FEATURE_TYPE_UNSPECIFIED', 'SECURE_BOOT']
    guest_os_features = set(features_enum_type.names())
    guest_os_features.difference_update(excluded_enums)
    if supported_features:
        guest_os_features.intersection_update(supported_features)
    if not guest_os_features:
        return
    parser.add_argument('--guest-os-features', metavar='GUEST_OS_FEATURE', type=arg_parsers.ArgList(element_type=lambda x: x.upper(), choices=sorted(guest_os_features)), help='      Enables one or more features for VM instances that use the\n      image for their boot disks. See the descriptions of supported features at:\n      https://cloud.google.com/compute/docs/images/create-delete-deprecate-private-images#guest-os-features.')