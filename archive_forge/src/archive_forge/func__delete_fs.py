from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def _delete_fs(module, blade):
    """In module Delete Filesystem"""
    api_version = blade.api_version.list_versions().versions
    if NFSV4_API_VERSION in api_version:
        if MULTIPROTOCOL_API_VERSION in api_version:
            blade.file_systems.update_file_systems(name=module.params['name'], attributes=FileSystem(nfs=NfsRule(v3_enabled=False, v4_1_enabled=False), smb=ProtocolRule(enabled=False), http=ProtocolRule(enabled=False), multi_protocol=MultiProtocolRule(access_control_style='shared'), destroyed=True))
        else:
            blade.file_systems.update_file_systems(name=module.params['name'], attributes=FileSystem(nfs=NfsRule(v3_enabled=False, v4_1_enabled=False), smb=ProtocolRule(enabled=False), http=ProtocolRule(enabled=False), destroyed=True))
    else:
        blade.file_systems.update_file_systems(name=module.params['name'], attributes=FileSystem(nfs=NfsRule(enabled=False), smb=ProtocolRule(enabled=False), http=ProtocolRule(enabled=False), destroyed=True))
    blade.file_systems.delete_file_systems(module.params['name'])