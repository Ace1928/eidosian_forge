from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as core_resources
def _ApplyPatch(self, authorized_orgs_desc_ref, authorized_orgs_desc, update_mask):
    """Applies a PATCH to the provided Authorized Orgs Desc."""
    m = self.messages
    request_type = m.AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsPatchRequest
    request = request_type(authorizedOrgsDesc=authorized_orgs_desc, name=authorized_orgs_desc_ref.RelativeName(), updateMask=','.join(update_mask))
    operation = self.client.accessPolicies_authorizedOrgsDescs.Patch(request)
    poller = util.OperationPoller(self.client.accessPolicies_authorizedOrgsDescs, self.client.operations, authorized_orgs_desc_ref)
    operation_ref = core_resources.REGISTRY.Parse(operation.name, collection='accesscontextmanager.operations')
    return waiter.WaitFor(poller, operation_ref, 'Waiting for PATCH operation [{}]'.format(operation_ref.Name()))