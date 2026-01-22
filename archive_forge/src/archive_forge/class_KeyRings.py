from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import resources
class KeyRings(base.Group):
    """Create and manage keyrings.

  A keyring is a toplevel logical grouping of keys.
  """
    category = base.IDENTITY_AND_SECURITY_CATEGORY

    @staticmethod
    def Args(parser):
        parser.display_info.AddUriFunc(cloudkms_base.MakeGetUriFunc(flags.KEY_RING_COLLECTION))