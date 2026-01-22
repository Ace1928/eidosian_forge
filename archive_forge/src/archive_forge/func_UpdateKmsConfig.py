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
def UpdateKmsConfig(self, kmsconfig_ref, kms_config, update_mask):
    """Send a Patch request for the Cloud NetApp KMS Config."""
    update_request = self.messages.NetappProjectsLocationsKmsConfigsPatchRequest(kmsConfig=kms_config, name=kmsconfig_ref.RelativeName(), updateMask=update_mask)
    update_op = self.client.projects_locations_kmsConfigs.Patch(update_request)
    return update_op