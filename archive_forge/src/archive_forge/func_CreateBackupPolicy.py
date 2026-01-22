from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateBackupPolicy(self, backuppolicy_ref, async_, backup_policy):
    """Create a Cloud NetApp Backup Policy."""
    request = self.messages.NetappProjectsLocationsBackupPoliciesCreateRequest(parent=backuppolicy_ref.Parent().RelativeName(), backupPolicyId=backuppolicy_ref.Name(), backupPolicy=backup_policy)
    create_op = self.client.projects_locations_backupPolicies.Create(request)
    if async_:
        return create_op
    operation_ref = resources.REGISTRY.ParseRelativeName(create_op.name, collection=constants.OPERATIONS_COLLECTION)
    return self.WaitForOperation(operation_ref)