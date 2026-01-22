from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def ConfigureManagement(self, registration_ref, transfer_lock, preferred_renewal_method):
    """Updates management settings.

    Args:
      registration_ref: a Resource reference to a
        domains.projects.locations.registrations resource.
      transfer_lock: The transfer lock state.
      preferred_renewal_method: The preferred Renewal Method.

    Returns:
      Operation: the long running operation to configure management
        registration.
    """
    management_settings = self.messages.ManagementSettings(transferLockState=transfer_lock, preferredRenewalMethod=preferred_renewal_method)
    updated_list = []
    if transfer_lock:
        updated_list += ['transfer_lock_state']
    if preferred_renewal_method:
        updated_list += ['preferred_renewal_method']
    update_mask = ','.join(updated_list)
    req = self.messages.DomainsProjectsLocationsRegistrationsConfigureManagementSettingsRequest(registration=registration_ref.RelativeName(), configureManagementSettingsRequest=self.messages.ConfigureManagementSettingsRequest(managementSettings=management_settings, updateMask=update_mask))
    return self._service.ConfigureManagementSettings(req)