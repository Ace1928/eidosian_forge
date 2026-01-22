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
def ParseSubjectFlags(args, is_ca):
    """Parses subject flags into a subject config.

  Args:
    args: The parser that contains all the flag values
    is_ca: Whether to parse this subject as a CA or not.

  Returns:
    A subject config representing the parsed flags.
  """
    messages = privateca_base.GetMessagesModule('v1')
    subject_config = messages.SubjectConfig(subject=messages.Subject(), subjectAltName=messages.SubjectAltNames())
    if args.IsSpecified('subject'):
        subject_config.subject = ParseSubject(args)
    if SanFlagsAreSpecified(args):
        subject_config.subjectAltName = ParseSanFlags(args)
    ValidateSubjectConfig(subject_config, is_ca=is_ca)
    return subject_config