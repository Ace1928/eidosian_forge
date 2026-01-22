from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.command_lib.iam import iam_util
def GetByDomain(self, domain):
    """Returns an Organization resource identified by the domain name.

    If no organization is returned, or if more than one organization is
    returned, this method will return None.

    Args:
      domain: A string representing an organizations associated domain.
              e.g. 'example.com'

    Returns:
      An instance of Organization or None if a unique organization cannot be
      determined.
    """
    domain_filter = 'domain:{0}'.format(domain)
    try:
        orgs_list = list(self.List(filter_=domain_filter))
    except exceptions.HttpBadRequestError:
        return None
    if len(orgs_list) == 1:
        return orgs_list[0]
    else:
        return None