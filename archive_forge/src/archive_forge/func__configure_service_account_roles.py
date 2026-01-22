from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.events import iam_util
from googlecloudsdk.api_lib.kuberun.core import events_constants
from googlecloudsdk.command_lib.events import exceptions
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _configure_service_account_roles(sa_config, gsa_emails):
    """Configures a service account with necessary iam roles for eventing."""
    log.status.Print('Configuring service account for {}.'.format(sa_config.description))
    service_account_ref = resources.REGISTRY.Parse(gsa_emails[sa_config].email, params={'projectsId': '-'}, collection=core_iam_util.SERVICE_ACCOUNTS_COLLECTION)
    should_bind_roles = gsa_emails[sa_config].is_default
    iam_util.PrintOrBindMissingRolesWithPrompt(service_account_ref, sa_config.recommended_roles, should_bind_roles)