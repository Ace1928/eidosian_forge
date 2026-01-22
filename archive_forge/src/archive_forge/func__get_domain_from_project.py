from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def _get_domain_from_project(self, project_ref):
    """Create a domain ref from a project ref.

        Based on the provided project ref, create a domain ref, so that the
        result can be returned in response to a domain API call.
        """
    if not project_ref['is_domain']:
        LOG.error('Asked to convert a non-domain project into a domain - Domain: %(domain_id)s, Project ID: %(id)s, Project Name: %(project_name)s', {'domain_id': project_ref['domain_id'], 'id': project_ref['id'], 'project_name': project_ref['name']})
        raise exception.DomainNotFound(domain_id=project_ref['id'])
    domain_ref = project_ref.copy()
    for k in ['parent_id', 'domain_id', 'is_domain', 'extra']:
        domain_ref.pop(k, None)
    return domain_ref