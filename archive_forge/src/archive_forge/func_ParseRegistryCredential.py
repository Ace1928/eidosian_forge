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
def ParseRegistryCredential(path, messages=None):
    messages = messages or devices.GetMessagesModule()
    contents = _ReadKeyFileFromPath(path)
    format_enum = messages.PublicKeyCertificate.FormatValueValuesEnum
    return messages.RegistryCredential(publicKeyCertificate=messages.PublicKeyCertificate(certificate=contents, format=format_enum.X509_CERTIFICATE_PEM))