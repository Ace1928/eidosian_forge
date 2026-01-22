from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.billing import utils
Link the given account to the given project.

    Args:
      project_ref: a Resource reference to the project to be linked to
      account_ref: a Resource reference to the account to link, or None to
        unlink the project from its current account.

    Returns:
      ProjectBillingInfo, the new ProjectBillingInfo
    