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
def EncryptKmsConfig(self, kmsconfig_ref, async_):
    """Encrypts the volumes attached to the Cloud NetApp KMS Config."""
    request = self.messages.NetappProjectsLocationsKmsConfigsEncryptRequest(name=kmsconfig_ref.RelativeName())
    encrypt_op = self.client.projects_locations_kmsConfigs.Encrypt(request)
    if async_:
        return encrypt_op
    operation_ref = resources.REGISTRY.ParseRelativeName(encrypt_op.name, collection=constants.OPERATIONS_COLLECTION)
    return self.WaitForOperation(operation_ref)