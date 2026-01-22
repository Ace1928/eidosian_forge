from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.backend_services import (
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import cdn_flags_utils as cdn_flags
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import reference_utils
from googlecloudsdk.command_lib.compute import signed_url_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
from googlecloudsdk.command_lib.compute.security_policies import (
from googlecloudsdk.core import log
def AddIapFlag(parser):
    flags.AddIap(parser, help="      Change the Identity Aware Proxy (IAP) service configuration for the\n      backend service. You can set IAP to 'enabled' or 'disabled', or modify\n      the OAuth2 client configuration (oauth2-client-id and\n      oauth2-client-secret) used by IAP. If any fields are unspecified, their\n      values will not be modified. For instance, if IAP is enabled,\n      '--iap=disabled' will disable IAP, and a subsequent '--iap=enabled' will\n      then enable it with the same OAuth2 client configuration as the first\n      time it was enabled. See\n      https://cloud.google.com/iap/ for more information about this feature.\n      ")