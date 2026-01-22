from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def GetDetailedHelpForSetIamPolicy(collection, example_id='', example_see_more='', additional_flags='', use_an=False):
    """Returns a detailed_help for a set-iam-policy command.

  Args:
    collection: Name of the command collection (ex: "project", "dataset")
    example_id: Collection identifier to display in a sample command
        (ex: "my-project", '1234')
    example_see_more: Optional "See ... for details" message. If not specified,
      includes a default reference to IAM managing-policies documentation
    additional_flags: str, additional flags to include in the example command
      (after the command name and before the ID of the resource).
     use_an: If True, uses "an" instead of "a" for the article preceding uses of
       the collection.

  Returns:
    a dict with boilerplate help text for the set-iam-policy command
  """
    if not example_id:
        example_id = 'example-' + collection
    if not example_see_more:
        example_see_more = '\n          See https://cloud.google.com/iam/docs/managing-policies for details\n          of the policy file format and contents.'
    additional_flags = additional_flags + ' ' if additional_flags else ''
    a = 'an' if use_an else 'a'
    return {'brief': 'Set IAM policy for {0} {1}.'.format(a, collection), 'DESCRIPTION': '{description}', 'EXAMPLES': textwrap.dedent("          The following command will read an IAM policy from 'policy.json' and\n          set it for {a} {collection} with '{id}' as the identifier:\n\n            $ {{command}} {flags}{id} policy.json\n\n          {see_more}".format(collection=collection, id=example_id, see_more=example_see_more, flags=additional_flags, a=a))}