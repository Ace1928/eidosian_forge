from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import ipaddress
import re
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import preset_profiles
from googlecloudsdk.command_lib.privateca import text_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from that bucket.
def ParseKeySpec(args):
    """Parses a specified KMS key version or algorithm to get a KeyVersionSpec."""
    messages = privateca_base.GetMessagesModule('v1')
    if args.IsSpecified('kms_key_version'):
        kms_key_version_ref = args.CONCEPTS.kms_key_version.Parse()
        return messages.KeyVersionSpec(cloudKmsKeyVersion=kms_key_version_ref.RelativeName())
    return messages.KeyVersionSpec(algorithm=_KEY_ALGORITHM_MAPPER.GetEnumForChoice(args.key_algorithm))