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
def GetDetailedHelpForAddIamPolicyBinding(collection, example_id, role='roles/editor', use_an=False, condition=False):
    """Returns a detailed_help for an add-iam-policy-binding command.

  Args:
    collection: Name of the command collection (ex: "project", "dataset")
    example_id: Collection identifier to display in a sample command
        (ex: "my-project", '1234')
    role: The sample role to use in the documentation. The default of
      'roles/editor' is usually sufficient, but if your command group's users
      would more likely use a different role, you can override it here.
    use_an: If True, uses "an" instead of "a" for the article preceding uses of
      the collection.
    condition: If True, add help text for condition.

  Returns:
    a dict with boilerplate help text for the add-iam-policy-binding command
  """
    a = 'an' if use_an else 'a'
    note = 'See https://cloud.google.com/iam/docs/managing-policies for details of policy role and principal types.'
    detailed_help = {'brief': 'Add IAM policy binding for {0} {1}.'.format(a, collection), 'DESCRIPTION': '{description}', 'EXAMPLES': "To add an IAM policy binding for the role of `{role}` for the user\n`test-user@gmail.com` on {a} {collection} with identifier\n`{example_id}`, run:\n\n  $ {{command}} {example_id} --member='user:test-user@gmail.com' --role='{role}'\n\nTo add an IAM policy binding for the role of `{role}` to the service\naccount `test-proj1@example.domain.com`, run:\n\n  $ {{command}} {example_id} --member='serviceAccount:test-proj1@example.domain.com' --role='{role}'\n\nTo add an IAM policy binding for the role of `{role}` for all\nauthenticated users on {a} {collection} with identifier\n`{example_id}`, run:\n\n  $ {{command}} {example_id} --member='allAuthenticatedUsers' --role='{role}'\n  ".format(collection=collection, example_id=example_id, role=role, a=a)}
    if condition:
        detailed_help['EXAMPLES'] = detailed_help['EXAMPLES'] + '\n\nTo add an IAM policy binding that expires at the end of the year 2018 for the\nrole of `{role}` and the user `test-user@gmail.com` on {a} {collection} with\nidentifier `{example_id}`, run:\n\n  $ {{command}} {example_id} --member=\'user:test-user@gmail.com\' --role=\'{role}\' --condition=\'expression=request.time < timestamp("2019-01-01T00:00:00Z"),title=expires_end_of_2018,description=Expires at midnight on 2018-12-31\'\n  '.format(collection=collection, example_id=example_id, role=role, a=a)
    detailed_help['EXAMPLES'] = '\n'.join([detailed_help['EXAMPLES'], note])
    return detailed_help