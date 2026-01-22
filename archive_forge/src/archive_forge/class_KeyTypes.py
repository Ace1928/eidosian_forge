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
class KeyTypes(enum.Enum):
    """Valid key types for device credentials."""
    RS256 = 1
    ES256 = 2
    RSA_PEM = 3
    RSA_X509_PEM = 4
    ES256_PEM = 5
    ES256_X509_PEM = 6

    def __init__(self, value):
        self.choice_name = self.name.replace('_', '-').lower()