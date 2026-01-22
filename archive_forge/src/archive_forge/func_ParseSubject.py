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
def ParseSubject(args):
    """Parses a dictionary with subject attributes into a API Subject type.

  Args:
    args: The argparse namespace that contains the flag values.

  Returns:
    Subject: the Subject type represented in the api.
  """
    subject_args = args.subject
    remap_args = {'CN': 'commonName', 'C': 'countryCode', 'ST': 'province', 'L': 'locality', 'O': 'organization', 'OU': 'organizationalUnit'}
    mapped_args = {}
    for key, val in subject_args.items():
        if key in remap_args:
            mapped_args[remap_args[key]] = val
        else:
            mapped_args[key] = val
    try:
        return messages_util.DictToMessageWithErrorCheck(mapped_args, privateca_base.GetMessagesModule('v1').Subject)
    except messages_util.DecodeError:
        raise exceptions.InvalidArgumentException('--subject', 'Unrecognized subject attribute.')