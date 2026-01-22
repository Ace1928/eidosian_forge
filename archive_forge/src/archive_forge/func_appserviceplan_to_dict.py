from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def appserviceplan_to_dict(plan):
    return dict(id=plan.id, name=plan.name, kind=plan.kind, location=plan.location, reserved=plan.reserved, is_linux=plan.reserved, provisioning_state=plan.provisioning_state, tags=plan.tags if plan.tags else None)