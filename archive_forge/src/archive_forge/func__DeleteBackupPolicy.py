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
def _DeleteBackupPolicy(self, async_, request):
    delete_op = self.client.projects_locations_backupPolicies.Delete(request)
    if async_:
        return delete_op
    operation_ref = resources.REGISTRY.ParseRelativeName(delete_op.name, collection=constants.OPERATIONS_COLLECTION)
    return self.WaitForOperation(operation_ref)