from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.command_lib.iam import iam_util
def WaitForLongOperation(operation):
    """Waits for the given google.longrunning.Operation to complete."""
    return dataplex_api.WaitForOperation(operation, dataplex_api.GetClientInstance().projects_locations_lakes, sleep_ms=10000, pre_start_sleep_ms=120000)