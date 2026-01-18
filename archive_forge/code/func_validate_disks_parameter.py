from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def validate_disks_parameter(self):
    errors = []
    create_option_reqs = [('import', ['source_uri', 'storage_account_id']), ('copy', ['source_uri']), ('empty', ['disk_size_gb'])]
    for disk in self.managed_disks:
        create_option = disk.get('create_option')
        for req in create_option_reqs:
            if create_option == req[0] and any((disk.get(opt) is None for opt in req[1])):
                errors.append('managed disk {0}/{1} has create_option set to {2} but not all required parameters ({3}) are set.'.format(disk.get('resource_group'), disk.get('name'), req[0], ','.join(req[1])))
    if errors:
        self.fail(msg='Some required options are missing from managed disks configuration.', errors=errors)