from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudiot import devices
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.command_lib.iot import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import times
import six
def _ValidatePublicKeyDict(public_key):
    unrecognized_keys = set(public_key.keys()) - set(_ALLOWED_KEYS)
    if unrecognized_keys:
        raise TypeError('Unrecognized keys [{}] for public key specification.'.format(', '.join(unrecognized_keys)))
    for key in _REQUIRED_KEYS:
        if key not in public_key:
            raise InvalidPublicKeySpecificationError('--public-key argument missing value for `{}`.'.format(key))