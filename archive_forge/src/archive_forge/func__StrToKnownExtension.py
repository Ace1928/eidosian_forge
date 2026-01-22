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
def _StrToKnownExtension(arg_name, val):
    trimmed_val = six.text_type(val).strip().lower()
    known_extensions = GetKnownExtensionMapping()
    if trimmed_val in known_extensions:
        return known_extensions[trimmed_val]
    else:
        raise exceptions.UnknownArgumentException(arg_name, 'expected one of [{}]'.format(','.join(known_extensions.keys())))