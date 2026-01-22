from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import policies as policies_api
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.meta import cache_util as meta_cache_util
from googlecloudsdk.command_lib.util import cache_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetDefaultPolicy():
    """Gets the ID of the default policy for the current account."""
    account = properties.VALUES.core.account.Get()
    if not account:
        log.info('Unable to automatically resolve policy since account property is not set.')
        return None
    domain = _GetDomain(account)
    if not domain:
        log.info('Unable to resolve domain for account [%s]', account)
        return None
    with meta_cache_util.GetCache('resource://', create=True) as cache:
        try:
            organization_ref = _GetOrganization(cache, domain)
            policy_ref = _GetPolicy(cache, organization_ref.RelativeName(), (organization_ref,))
        except DefaultPolicyResolutionError as err:
            log.info('Unable to automatically resolve policy: %s', err)
            return None
    return policy_ref.Name()