from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from six.moves import map  # pylint: disable=redefined-builtin
from the device. This flag can be specified multiple times to add multiple
def AddDeviceRegistryCredentialFlagsToParser(parser, credentials_surface=True):
    """Add device credential flags to arg parser."""
    help_text = 'Path to a file containing an X.509v3 certificate ([RFC5280](https://www.ietf.org/rfc/rfc5280.txt)), encoded in base64, and wrapped by `-----BEGIN CERTIFICATE-----` and `-----END CERTIFICATE-----`.'
    if not credentials_surface:
        base.Argument('--public-key-path', type=str, help=help_text).AddToParser(parser)
    else:
        base.Argument('--path', type=str, required=True, help=help_text).AddToParser(parser)