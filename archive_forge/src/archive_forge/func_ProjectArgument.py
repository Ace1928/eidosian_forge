from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers as resource_manager_completers
from googlecloudsdk.core import properties
def ProjectArgument(help_text_to_prepend=None, help_text_to_overwrite=None):
    """Creates project argument.

  Args:
    help_text_to_prepend: str, help text to prepend to the generic --project
      help text.
    help_text_to_overwrite: str, help text to overwrite the generic --project
      help text.

  Returns:
    calliope.base.Argument, The argument for project.
  """
    if help_text_to_overwrite:
        help_text = help_text_to_overwrite
    else:
        help_text = "The Google Cloud project ID to use for this invocation. If\nomitted, then the current project is assumed; the current project can\nbe listed using `gcloud config list --format='text(core.project)'`\nand can be set using `gcloud config set project PROJECTID`.\n\n`--project` and its fallback `{core_project}` property play two roles\nin the invocation. It specifies the project of the resource to\noperate on. It also specifies the project for API enablement check,\nquota, and billing. To specify a different project for quota and\nbilling, use `--billing-project` or `{billing_project}` property.\n    ".format(core_project=properties.VALUES.core.project, billing_project=properties.VALUES.billing.quota_project)
        if help_text_to_prepend:
            help_text = '\n\n'.join((help_text_to_prepend, help_text))
    return base.Argument('--project', metavar='PROJECT_ID', dest='project', category=base.COMMONLY_USED_FLAGS, suggestion_aliases=['--application'], completer=resource_manager_completers.ProjectCompleter, action=actions.StoreProperty(properties.VALUES.core.project), help=help_text)