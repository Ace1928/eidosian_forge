from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def CreateConfidentialInstanceMessage(messages, args, support_confidential_compute_type, support_confidential_compute_type_tdx):
    """Create confidentialInstanceConfig message for VM."""
    confidential_instance_config_msg = None
    enable_confidential_compute = None
    confidential_instance_type = None
    if hasattr(args, 'confidential_compute') and args.IsSpecified('confidential_compute') and isinstance(args.confidential_compute, bool):
        enable_confidential_compute = args.confidential_compute
    if support_confidential_compute_type and hasattr(args, 'confidential_compute_type') and args.IsSpecified('confidential_compute_type') and isinstance(args.confidential_compute_type, six.string_types):
        confidential_instance_type = messages.ConfidentialInstanceConfig.ConfidentialInstanceTypeValueValuesEnum(args.confidential_compute_type)
        if not support_confidential_compute_type_tdx and 'TDX' in messages.ConfidentialInstanceConfig.ConfidentialInstanceTypeValueValuesEnum:
            enable_confidential_compute = None
            confidential_instance_type = None
    if confidential_instance_type is not None:
        confidential_instance_config_msg = messages.ConfidentialInstanceConfig(confidentialInstanceType=confidential_instance_type)
    elif enable_confidential_compute is not None:
        confidential_instance_config_msg = messages.ConfidentialInstanceConfig(enableConfidentialCompute=enable_confidential_compute)
    return confidential_instance_config_msg