from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def vmextension_to_dict(extension):
    """
    Serializing the VM Extension from the API to Dict
    :return: dict
    """
    return dict(id=extension.id, name=extension.name, location=extension.location, publisher=extension.publisher, virtual_machine_extension_type=extension.type_properties_type, type_handler_version=extension.type_handler_version, auto_upgrade_minor_version=extension.auto_upgrade_minor_version, settings=extension.settings, protected_settings=extension.protected_settings)