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
def GetConfidentialVmType(args, support_confidential_compute_type):
    """Returns the CONFIDENTIAL_VM_TYPES of the machine."""
    confidential_vm_type = None
    if args.IsSpecified('confidential_compute') and args.confidential_compute:
        confidential_vm_type = constants.CONFIDENTIAL_VM_TYPES.SEV
    if support_confidential_compute_type and args.IsSpecified('confidential_compute_type') and (args.confidential_compute_type is not None):
        confidential_vm_type = getattr(constants.CONFIDENTIAL_VM_TYPES, args.confidential_compute_type.upper())
    return confidential_vm_type