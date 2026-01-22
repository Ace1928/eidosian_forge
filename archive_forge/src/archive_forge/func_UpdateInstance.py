from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.filestore.backups import util as backup_util
from googlecloudsdk.command_lib.filestore.snapshots import util as snapshot_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def UpdateInstance(self, instance_ref, instance_config, update_mask):
    """Send a Patch request for the Cloud Filestore instance."""
    update_request = self.messages.FileProjectsLocationsInstancesPatchRequest(instance=instance_config, name=instance_ref.RelativeName(), updateMask=update_mask)
    update_op = self.client.projects_locations_instances.Patch(update_request)
    return update_op