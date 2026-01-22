from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import exceptions
from googlecloudsdk.command_lib.compute.instances import flags
def _original_email(self, instance_ref, client):
    """Return email of service account instance is using."""
    instance = self._get_instance(instance_ref, client)
    if instance is None:
        return None
    orignal_service_accounts = instance.serviceAccounts
    if orignal_service_accounts:
        return orignal_service_accounts[0].email
    return None