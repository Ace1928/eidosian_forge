from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def UpdateStoragePool(self, storagepool_ref, storagepool_config, update_mask):
    """Send a Patch request for the Cloud NetApp Storage Pool."""
    update_request = self.messages.NetappProjectsLocationsStoragePoolsPatchRequest(storagePool=storagepool_config, name=storagepool_ref.RelativeName(), updateMask=update_mask)
    update_op = self.client.projects_locations_storagePools.Patch(update_request)
    return update_op